"""Schedule durable LeRobot exports outside the time-sensitive control process."""

import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def collection_publisher(root):
    root = Path(root)
    stopped = threading.Event()

    def publish():
        last_seen = frozenset()
        while True:
            closing = stopped.is_set()
            ids = frozenset(p.name for p in (root / "episodes").glob("*") if p.is_dir())
            if ids and ids != last_seen:
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "alohamini",
                            "rlt",
                            "export-v3",
                            "--output-dir",
                            str(root),
                        ],
                        stdin=subprocess.DEVNULL,
                        check=True,
                    )
                    last_seen = ids
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"v3 导出待重试，原始回合保存在 {root}（{exc}）。", flush=True)
                    if not closing:
                        stopped.wait(15)
            if closing:
                return
            stopped.wait(2)

    worker = threading.Thread(target=publish, name="RLT-v3-publisher", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
