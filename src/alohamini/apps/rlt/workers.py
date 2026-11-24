"""Spawned GPU services and a separate, freshness-checked passive leader reader."""

import multiprocessing
import queue
import threading
import time


def _inference_process(config, initial, jobs, results, stop):
    from alohamini.policies.rlt.policy import RLTPolicy

    try:
        policy = RLTPolicy(config)
        policy.bind(initial)
        native = policy.native
        results.put(
            dict(
                kind="ready",
                names=native.names,
                fps=native.fps,
                cameras=native.manifest["cameras"],
                fixed=native.fixed_dimensions,
                metadata=native.robot_metadata,
                base_contract=policy.contract,
                actor_identity=policy.actor_identity,
                gate_identity=policy.gate_identity,
            )
        )

        generation = None

        def predict(job):
            nonlocal generation
            if job["generation"] != generation:
                native.reset()
                generation = job["generation"]
            started = time.monotonic()
            prediction = policy.predict(
                job["snapshot"],
                observation_id=job["id"],
                apply_actor=job.get("critical", False),
            )
            results.put(
                dict(
                    kind="prediction",
                    generation=job["generation"],
                    id=job["id"],
                    sampled_s=job["snapshot"].request_started_s,
                    observation_sequence=getattr(job["snapshot"], "payload", {})
                    .get("_host_timing", {})
                    .get("state_sequence"),
                    inference_s=time.monotonic() - started,
                    **prediction,
                ),
                timeout=1,
            )

        while not stop.is_set():
            try:
                predict(jobs.get(timeout=0.05))
            except queue.Empty:
                continue
    except InterruptedError:
        return
    except Exception as exc:
        results.put(dict(kind="error", error=f"{type(exc).__name__}: {exc}"), timeout=1)
        raise


class InferenceWorker:
    def __init__(self, config, initial):
        context = multiprocessing.get_context("spawn")
        self.jobs = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=4)
        self.stop = context.Event()
        self.process = context.Process(
            target=_inference_process,
            args=(config, initial, self.jobs, self.results, self.stop),
            name="RLTInference",
        )
        self.pending = False
        self.process.start()

    def submit(self, job):
        if self.pending:
            return False
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            return False
        self.pending = True
        return True

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError(f"Inference process exited ({self.process.exitcode})") from None
            return None
        if result["kind"] == "error":
            raise RuntimeError(result["error"])
        if result["kind"] == "prediction":
            self.pending = False
        return result

    def close(self):
        self.stop.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        for channel in (self.jobs, self.results):
            channel.cancel_join_thread()
            channel.close()


class LeaderReader:
    """Only this thread touches the serial buses. A blocked read expires naturally."""

    def __init__(self, config, units):
        self.config, self.units = config, units
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="RLTLeader", daemon=True)
        self.thread.start()

    def _run(self):
        from alohamini.hardware.leader import BimanualLeader

        try:
            with BimanualLeader(
                self.config.robot_model,
                leader_id=self.config.leader_id,
                calibration_dir=self.config.calibration_dir,
                left_port=self.config.left_port,
                active_sides=("left",),
                interactive_calibration=False,
            ) as leader:
                while not self.stop.is_set():
                    started = time.monotonic()
                    action = leader.read(self.units)
                    with self._lock:
                        self.latest = (started, action)
                    self.stop.wait(max(0, 0.01 - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc
            print(f"主臂读取服务不可用，请检查串口和已有标定。{exc}", flush=True)

    def read(self, max_age_s):
        if self.error is not None:
            return None
        with self._lock:
            value = self.latest
        if value is None or time.monotonic() - value[0] > max_age_s:
            return None
        return value

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
