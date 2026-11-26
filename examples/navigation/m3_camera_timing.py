"""Convert Pi exposure timestamps to the workstation clock before publishing."""
import sys
import threading
import time

from m3_clock_check import assess, measure


class CameraClock:
    """Replace the complete anchor on a good probe. Expiry is checked on every frame."""

    def __init__(self):
        self.anchor = None

    def refresh(self, samples, now):
        result = assess(samples, bound_ns=None)
        lo, hi = result["pi_minus_pc_ns"]
        details = (f"lo_ns={lo} hi_ns={hi} width_ms={(hi - lo) / 1e6:.3f} "
                   f"min_rtt_ms={result['min_rtt_ns'] / 1e6:.3f} "
                   f"used={result['used']}/{len(samples)}")
        if not result["passed"]:
            raise ValueError(f"{result['reason']}: {details}")
        # Taking the upper bound prevents dating an exposure later than it occurred.
        self.anchor = (now, hi)
        return details

    def corrected_stamp(self, capture_ns, now, wall_ns):
        anchor = self.anchor
        if anchor is None:
            raise ValueError("no valid Pi clock anchor")
        if now - anchor[0] > 30:
            raise ValueError("Pi clock anchor older than 30 s")
        corrected = capture_ns - anchor[1]
        age_ns = wall_ns - corrected
        if not 0 <= age_ns < 250_000_000:
            raise ValueError(f"camera outside capture deadline: age_ms={age_ns / 1e6:.3f}")
        return corrected


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from alohamini_camera.camera_node import AlohaMiniCameraNode, ros_stamp_from_unix_ns

    ctl, peer = sys.argv.pop(1), sys.argv.pop(1)
    clock = CameraClock()
    done = threading.Event()

    def refresh():
        try:
            samples = measure(peer, ctl)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Pi clock probe failed: {exc}; lo_ns=n/a hi_ns=n/a min_rtt_ms=n/a",
                  flush=True)
            return
        try:
            details = clock.refresh(samples, time.monotonic())
            print(f"measured Pi-PC {details}", flush=True)
        except ValueError as exc:
            print(f"Pi clock probe failed: {exc}", flush=True)

    def probe():
        while not done.wait(3):
            refresh()

    class Timed(AlohaMiniCameraNode):
        def frame_stamp(self, frame):
            stamp = clock.corrected_stamp(frame.capture_unix_ns, time.monotonic(), time.time_ns())
            return ros_stamp_from_unix_ns(stamp)

    # Probe before subscribing so startup does not discard frames during the SSH handshake.
    refresh()
    rclpy.init()
    node = Timed()
    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        done.set()
        print(f"camera summary: received={node.received} rejected={node.invalid}", flush=True)
        node.destroy_node()
        rclpy.try_shutdown()
        worker.join(timeout=8)


if __name__ == "__main__":
    main()
