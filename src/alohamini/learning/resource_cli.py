"""JSON-only command line for inspecting a training configuration's resources."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from alohamini.learning.resource_checks import ResourceChecks


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect training resources and print a JSON report"
    )
    parser.add_argument(
        "--config",
        "--config_path",
        required=True,
        type=Path,
        help="Training JSON configuration; no dataset or output directory is required",
    )
    args = parser.parse_args(argv)
    try:
        from alohamini.learning.resources import check_training_resources

        settings = json.loads(args.config.expanduser().read_text())
        with redirect_stdout(sys.stderr):
            report = check_training_resources(settings)
    except (OSError, ValueError, ImportError) as error:
        checks = ResourceChecks(None, strict=False)
        checks.fail("configuration", error)
        report = checks.report()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
