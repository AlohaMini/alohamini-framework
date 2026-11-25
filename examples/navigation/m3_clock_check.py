"""Read-only host_wall preflight. Use a trusted SSH host key or an existing master."""
import argparse
import json
import select
import subprocess
import time


def assess(samples, bound_ns=2_000_000):
    """Intersect clock-offset intervals. Sign is Pi wall minus workstation wall."""
    if len(samples) < 5:
        raise ValueError("at least five exchanges required")
    intervals = []
    for sent, remote, received, steady_elapsed in samples:
        elapsed = received - sent
        if elapsed < 0 or steady_elapsed < 0 or abs(elapsed - steady_elapsed) > 250_000:
            raise ValueError("local wall clock changed during exchange")
        intervals.append((remote - received, remote - sent))
    lo = max(pair[0] for pair in intervals)
    hi = min(pair[1] for pair in intervals)
    return dict(pi_minus_pc_ns=[lo, hi], min_rtt_ns=min(r-s for s, _, r, _ in samples),
                passed=lo <= hi and max(abs(lo), abs(hi)) <= bound_ns, bound_ns=bound_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pi")
    parser.add_argument("--ssh-control")
    args = parser.parse_args()
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=4"]
    if args.ssh_control:
        command += ["-S", args.ssh_control]
    # The startup handshake is outside all timed exchanges.
    command += [args.pi, """python3 -u -c 'import sys,time; print("ready"); [print(time.time_ns()) for line in sys.stdin]'"""]
    samples = []
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0) as proc:
        try:
            def receive(timeout):
                if not select.select([proc.stdout], [], [], timeout)[0]:
                    raise TimeoutError("timed out waiting for Pi clock")
                return proc.stdout.readline().strip()
            if receive(8) != b"ready":
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
    result = assess(samples)
    print(json.dumps({**result, "samples": samples}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

