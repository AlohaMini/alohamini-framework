"""Atomic artifacts and asynchronous episode recording with explicit reward commits."""

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from uuid import uuid4

from alohamini.protocol import HostSnapshot
from alohamini.rlt.control import feedback_after_command
from alohamini.schema import CommandIdentity


def _digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def digest_file(path):
    with Path(path).open("rb") as stream:
        return _digest_stream(stream)


def _replace_artifact(temporary, path):
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 4:
                raise
            # Windows readers can briefly hold a file without delete sharing.
            time.sleep(0.01 * (attempt + 1))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    _replace_artifact(temporary, path)


class RunLock:
    """OS advisory lock is released on process exit, including a crash."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        self.stream.write(b"0")
        self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError(f"Another process holds {self.path}") from None
        return self

    def __exit__(self, *_exc):
        self.stream.close()


def save_snapshot(path, snapshot):
    path = Path(path)
    path.mkdir(parents=True)
    atomic_json(
        path / "state.json",
        dict(
            payload=snapshot.payload,
            request_started_s=snapshot.request_started_s,
            received_s=snapshot.received_s,
        ),
    )
    for index, data in enumerate(snapshot.images.values()):
        # Names stay in JSON. User-supplied camera names never become paths.
        (path / f"{index}.jpg").write_bytes(data)
    atomic_json(path / "cameras.json", list(snapshot.images))


def load_snapshot(path):
    path = Path(path)
    data = json.loads((path / "state.json").read_text(encoding="utf-8"))
    cameras = json.loads((path / "cameras.json").read_text(encoding="utf-8"))
    return HostSnapshot(
        **data,
        images={
            camera: (path / f"{index}.jpg").read_bytes() for index, camera in enumerate(cameras)
        },
    )


def json_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recording_history(root, *, resume):
    """Check append intent under the control lock and read committed history."""
    root = Path(root)
    contract = root / "run-contract.json"
    existing = contract.exists() or (root / "run-config.json").exists()
    existing = existing or any(
        directory.exists() and any(directory.iterdir())
        for directory in (root / "episodes", root / "pending")
    )
    if existing and not resume:
        raise ValueError(f"Recording directory already has a run. Add --resume to continue: {root}")
    if existing and not contract.is_file():
        raise ValueError(f"Existing recording is missing run-contract.json: {root}")
    history = []
    for directory in sorted((root / "episodes").glob("*")):
        if not directory.is_dir():
            continue
        summary = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
        if (
            not isinstance(summary, dict)
            or summary.get("episode_id") != directory.name
            or summary.get("status") != "committed"
            or type(summary.get("reward")) is not int
            or summary["reward"] not in (0, 1)
            or type(summary.get("takeovers", 0)) is not int
            or summary.get("takeovers", 0) < 0
        ):
            raise ValueError(f"Invalid committed episode metadata: {directory}")
        summary.setdefault("takeovers", 0)
        history.append(summary)
    return history


def validate_training_episode(directory, summary):
    if not summary.get("training_valid", True):
        raise ValueError(f"Episode is excluded from training: {summary.get('invalid_reason')}")
    if not summary.get("requires_acknowledgements") and "started_at" not in summary:
        return
    outcomes = {
        CommandIdentity(**row["command"]): row for row in json_lines(directory / "acks.jsonl")
    }
    names = set(summary["base_contract"]["learned_names"])
    dispositions = summary.get("requires_command_dispositions", False)
    whole_episode = summary.get("phase_labels") == "optional-operator-v2" and (
        summary.get("critical_entries") == 0
    )
    for row in json_lines(directory / "steps.jsonl"):
        if (row["critical"] or whole_episode) and row["source"] in ("VLA", "RLT", "HUMAN"):
            outcome = outcomes.get(CommandIdentity(**row["command"]), {})
            status = outcome.get("status", "applied")
            if dispositions and status in ("superseded", "rejected"):
                continue
            if status != "applied" or not names <= outcome.get("targets", {}).keys():
                raise ValueError(
                    f"Episode is still awaiting Host acceptance for step {row['index']}"
                )
    data = json.loads((directory / "terminal/state.json").read_text(encoding="utf-8"))
    terminal = HostSnapshot(**data, images={})
    if dispositions:
        fence = outcomes.get(CommandIdentity(**summary["end_fence"]), {})
        if fence.get("status") != "applied":
            raise ValueError("Final HOLD has no application proof")
        for outcome in outcomes.values():
            if outcome.get("status", "applied") == "applied" and not feedback_after_command(
                terminal, outcome
            ):
                raise ValueError("Terminal feedback predates a relevant command application")
    else:
        for outcome in outcomes.values():
            observed = outcome.get("observed_s", outcome.get("time"))
            if observed is None or terminal.request_started_s <= observed:
                raise ValueError("Legacy terminal has no post-application ordering proof")


def commit_episode(directory, reward):
    """Called only after the writer closed an explicitly ended episode."""
    if type(reward) is not int or reward not in (0, 1):
        raise ValueError("Reward must be the integer 0 or 1")
    directory = Path(directory).absolute()
    root = directory.parent.parent.resolve()
    target = directory.parent.parent / "episodes" / directory.name
    if not directory.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
        raise ValueError("Episode commit paths must remain within the run directory")
    if not directory.exists() and target.exists():
        existing = json.loads((target / "episode.json").read_text(encoding="utf-8"))
        if existing.get("reward") != reward:
            raise ValueError("Episode is already committed with a different reward")
        return target
    summary = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    valid = summary.get("training_valid", True)
    if not summary.get("ended") or (valid and not (directory / "terminal/state.json").is_file()):
        raise ValueError("An interrupted episode has no valid terminal boundary to score")
    if summary.get("writer_error"):
        raise ValueError("Episode recording failed and cannot be committed")
    if valid:
        validate_training_episode(directory, summary)
    summary.update(reward=reward, status="committed", committed_at=time.time())
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        existing = json.loads((target / "episode.json").read_text(encoding="utf-8"))
        if existing != summary:
            raise ValueError("Episode ID is already committed")
        return target
    atomic_json(directory / "episode.json", summary)
    for attempt in range(5):
        try:
            directory.rename(target)
            break
        except PermissionError:
            if os.name != "nt" or attempt == 4:
                raise
            # Windows indexers can briefly retain a just-closed episode file.
            time.sleep(0.025 * 2**attempt)
    return target


class EpisodeRecorder:
    """One bounded disk worker. Slow storage never blocks the control loop."""

    def __init__(self, root, metadata):
        self.id = uuid4().hex
        self.directory = Path(root) / "pending" / self.id
        self.directory.mkdir(parents=True)
        self.summary = dict(metadata, episode_id=self.id, status="pending", ended=False)
        atomic_json(self.directory / "episode.json", self.summary)
        self.error = None
        self.committed = None
        self._queue = queue.Queue(maxsize=128)
        self._done = threading.Event()
        self.terminal_saved = threading.Event()
        self._thread = threading.Thread(target=self._run, name="RLTEpisodeWriter", daemon=True)
        self._thread.start()

    def put(self, kind, value):
        self.check()
        try:
            self._queue.put_nowait((kind, value))
        except queue.Full:
            self.error = RuntimeError("RLT recording queue is full, episode remains pending")
            raise self.error from None

    def step(self, index, snapshot, record):
        self.put("step", (index, snapshot, record))

    def acknowledgement(self, command):
        self.put("ack", command)

    def end_episode(self, snapshot, summary):
        self.put("end", (snapshot, summary))

    def confirm_reward(self, reward):
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("Reward must be the integer 0 or 1")
        self.put("summary", dict(reward_requested=reward))

    def submit_reward(self, reward):
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("Reward must be the integer 0 or 1")
        self.put("reward", reward)

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Episode writer failed: {self.error}") from self.error

    def _run(self):
        try:
            with (
                (self.directory / "steps.jsonl").open("a", encoding="utf-8") as steps,
                (self.directory / "acks.jsonl").open("a", encoding="utf-8") as acks,
            ):
                while True:
                    kind, value = self._queue.get()
                    if kind == "close":
                        break
                    if self.error is not None:
                        continue
                    if kind == "step":
                        index, snapshot, record = value
                        save_snapshot(self.directory / "observations" / str(index), snapshot)
                        steps.write(json.dumps(dict(record, index=index), allow_nan=False) + "\n")
                    elif kind == "ack":
                        acks.write(json.dumps(value, allow_nan=False) + "\n")
                    elif kind == "end":
                        snapshot, summary = value
                        steps.flush()
                        acks.flush()
                        os.fsync(steps.fileno())
                        os.fsync(acks.fileno())
                        if snapshot is not None:
                            save_snapshot(self.directory / "terminal", snapshot)
                        self.summary.update(summary, ended=True)
                        atomic_json(self.directory / "episode.json", self.summary)
                        self.terminal_saved.set()
                    elif kind == "summary":
                        self.summary.update(value)
                        atomic_json(self.directory / "episode.json", self.summary)
                    elif kind == "reward":
                        steps.flush()
                        acks.flush()
                        os.fsync(steps.fileno())
                        os.fsync(acks.fileno())
                        break
            if kind == "reward" and self.error is None:
                self.committed = commit_episode(self.directory, value)
        except Exception as exc:
            self.error = exc
        finally:
            if self.error is not None:
                self.summary["writer_error"] = str(self.error)
                try:
                    atomic_json(self.directory / "episode.json", self.summary)
                except OSError:
                    pass  # The original disk error remains observable through check().
            self._done.set()

    def close(self):
        if not self._done.is_set():
            # Called during shutdown after the Host hold request.
            deadline = time.monotonic() + 10
            while self._thread.is_alive() and time.monotonic() < deadline:
                try:
                    self._queue.put(("close", None), timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join(timeout=max(0, deadline - time.monotonic()))
            if self._thread.is_alive():
                raise RuntimeError("Episode writer did not drain; recovery files remain pending")
        self.check()
