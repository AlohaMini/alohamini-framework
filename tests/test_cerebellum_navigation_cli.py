import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from alohamini.cerebellum import cli


class NavigationCLI(unittest.TestCase):
    def test_parser_and_top_level_dispatch(self):
        from alohamini.cli import main
        args = cli.parser().parse_args(["run", "--instruction", "go left"])
        self.assertFalse(args.chair)
        self.assertFalse(args.dry_run)
        self.assertEqual((args.ros_domain, str(args.pi_host), args.color), (181, "192.168.2.2", None))
        with patch.object(cli, "main", return_value=7) as navigate:
            self.assertEqual(main(["navigate", "run", "--instruction", "go left"]), 7)
            navigate.assert_called_once_with(["run", "--instruction", "go left"])
        for action in ("preflight","drive","status","estop","stop","restart","start"):
            self.assertEqual(cli.parser().parse_args([action]).action, action)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux signal and process groups")
    def test_run_cleanup_signals_dry_run_and_nonzero_preflight(self):
        # The real CLI invokes an inert shell in place of the hardware launcher.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/"examples/navigation"
            folder.mkdir(parents=True)
            script = folder/"m3_navigation.sh"
            script.write_text('echo "$1 $M3_DRY_RUN $M3_SESSION_TOKEN $M3_STOP_TOKEN" >> "$TRACE"\n'
                              'if [[ "$1" == preflight && "$FAIL" == yes ]]; then exit 17; fi\n'
                              'if [[ "$1" == drive ]]; then exec sleep 60; fi\n')
            harness = ("import sys; from pathlib import Path; from alohamini.cerebellum import cli; "
                       "sys.exit(cli.execute(cli.parser().parse_args(sys.argv[2:]),Path(sys.argv[1])))")
            for idx, sig in enumerate((signal.SIGINT, signal.SIGTERM, None)):
                trace = root/f"trace-{idx}"
                args = ["run", "--instruction", "Go left", "--dry-run", "--chair", "--output-dir", str(root/f"out-{idx}")]
                with subprocess.Popen([sys.executable,"-c",harness,str(script),*args],
                                      env=dict(os.environ,TRACE=str(trace),FAIL="no" if sig else "yes")) as proc:
                    try:
                        if sig:
                            for _ in range(150):
                                if trace.exists() and "drive true" in trace.read_text():
                                    break
                                time.sleep(.02)
                            else:
                                self.fail("mock drive did not start")
                            proc.send_signal(sig)
                        self.assertEqual(proc.wait(timeout=8),128+sig if sig else 17)
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                lines=trace.read_text().splitlines()
                self.assertEqual([s.split()[0] for s in lines],
                                 ["start","preflight","drive","stop"] if sig else ["start","preflight","stop"])
                self.assertEqual(lines[-1].split()[2:], [lines[0].split()[2]]*2)
