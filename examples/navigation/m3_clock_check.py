"""Read-only SSH clock probe for raw host_wall checks and camera timestamp correction."""
import argparse
import json
import select
import subprocess
import time


MAX_UNCERTAINTY_NS = 5_000_000


def assess(samples, bound_ns=2_000_000):
    """Estimate Pi minus PC. A raw host_wall consumer also needs near-synchronous clocks."""
    if len(samples) < 5:
        raise ValueError("at least five exchanges required")
    selected = sorted(samples, key=lambda sample: sample[3])[:8] if bound_ns is None else samples
    lo = max(remote - received for _, remote, received, _ in selected)
    hi = min(remote - sent for sent, remote, _, _ in selected)
    min_rtt = min(sample[3] for sample in selected)
    for sent, _, received, elapsed in selected:
        if elapsed < 0 or received < sent or abs(received - sent - elapsed) > 250_000:
            raise ValueError(f"local wall clock changed during exchange: lo_ns={lo} hi_ns={hi} "
                             f"min_rtt_ms={min_rtt / 1e6:.3f}")
    reason = None
    if hi < lo:
        reason = "offset intervals do not intersect"
    elif hi - lo > MAX_UNCERTAINTY_NS:
        reason = "offset uncertainty exceeds 5 ms"
    elif bound_ns is not None and max(abs(lo), abs(hi)) > bound_ns:
        reason = "uncorrected host_wall offset exceeds bound"
    return dict(pi_minus_pc_ns=[lo, hi], min_rtt_ns=min_rtt, used=len(selected),
                passed=reason is None, reason=reason, bound_ns=bound_ns)


def measure(pi, ssh_control):
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=4"]
    if ssh_control:
        command += ["-S", ssh_control]
    command += [pi, """python3 -u -c 'import sys,time; print("ready"); [print(time.time_ns()) for line in sys.stdin]'"""]
    samples = []
    deadline = time.monotonic() + 7
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0) as proc:
        try:
            def receive(timeout):
                remaining = min(timeout, deadline - time.monotonic())
                if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
                    raise TimeoutError("timed out waiting for Pi clock")
                return proc.stdout.readline().strip()

            if receive(7) != b"ready":
                raise RuntimeError("clock probe handshake failed")
            for _ in range(16):
                start = time.monotonic_ns()
                sent = time.time_ns()
                proc.stdin.write(b"t\n")
                remote = int(receive(2))
                received = time.time_ns()
                samples.append((sent, remote, received, time.monotonic_ns() - start))
        finally:
            proc.stdin.close()
            proc.terminate()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pi")
    parser.add_argument("--ssh-control")
    parser.add_argument("--corrected", action="store_true",
                        help="assess uncertainty when the consumer subtracts the measured offset")
    args = parser.parse_args()
    samples = measure(args.pi, args.ssh_control)
    result = assess(samples, bound_ns=None) if args.corrected else assess(samples)
    print(json.dumps({**result, "samples": samples}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
