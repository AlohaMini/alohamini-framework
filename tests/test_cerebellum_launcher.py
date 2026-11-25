"""Exercise startup rollback and the one-command launcher without ROS or hardware."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux process groups and flock")
class LauncherTests(unittest.TestCase):
    def test_interrupt_loading_then_restart_and_cancel_before_go(self):
        source = Path(__file__).resolve().parents[1] / "examples/navigation/m3_navigation.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nav = root / "examples/navigation"
            nav.mkdir(parents=True)
            launcher = nav / source.name
            shutil.copyfile(source, launcher)
            (nav / "m2b_model_services.sh").write_text(
                'if [[ "$READY" == yes ]]; then\n'
                '  echo "READY host=" > "$M2B_LOG_DIR/lightnav.log"\n'
                '  echo "Metric3D ready" > "$M2B_LOG_DIR/depth.log"\n'
                'fi\nexec sleep 60\n')
            conda = root / "etc/profile.d/conda.sh"
            conda.parent.mkdir(parents=True)
            conda.write_text("conda() { :; }\n")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name, body in {
                "ssh": "exit 0", "ss": "exit 0", "nvidia-smi": "exit 0",
                "timeout": 'shift; if [[ "$1" == bash ]]; then exit 0; fi; exec "$@"',
                "ros2": 'echo "$*" >> "$FAKE_ROS_LOG"; if [[ "$1" == service ]]; then echo success=True; fi',
                "python": "exec sleep 60",
            }.items():
                path = bin_dir / name
                path.write_text("#!/bin/bash\n" + body + "\n")
                path.chmod(0o755)
            config = root / "config.json"
            config.write_text("{}")
            run = root / "logs"
            ros_log = root / "ros.log"
            env = dict(os.environ, CONDA_BASE=tmp, M3_CONFIG=str(config), M3_RUN_DIR=str(run),
                       ROS_DOMAIN_ID="181", M3_COLOR="", M3_TARGET_ID="", FAKE_ROS_LOG=str(ros_log),
                       PATH=f"{bin_dir}:{os.environ['PATH']}")
            def wait_until(check):
                for _ in range(150):
                    if check():
                        return
                    time.sleep(.03)
                self.fail("timed out waiting for launcher lifecycle transition")

            def assert_clean():
                for file in run.glob("*.pid"):
                    pid, _ = file.read_text().split()
                    wait_until(lambda: not Path(f"/proc/{pid}/cwd").exists())

            cmd = ["bash", str(launcher), "run", "Stop at the doorway."]
            with subprocess.Popen(cmd, env=dict(env, READY="no"), start_new_session=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                try:
                    wait_until(lambda: (run / "services.pid").is_file()
                               and (run / "services.pid").stat().st_size > 0)
                    os.killpg(proc.pid, signal.SIGINT)
                    output, _ = proc.communicate(timeout=10)
                    self.assertEqual(proc.returncode, 130, output)
                finally:
                    if proc.poll() is None:
                        proc.kill()
            assert_clean()
            result = subprocess.run(cmd, env=dict(env, READY="yes"), input="CANCEL\n",
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("Type GO", result.stdout)
            self.assertEqual((run / "instruction").read_text(), cmd[-1])
            self.assertNotIn("data: true", ros_log.read_text())
            assert_clean()
