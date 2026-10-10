"""Small, data-free checks for the reviewer-facing reproducibility package."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    run([sys.executable, "-c", "import numpy, torch, scipy, requests; print('imports OK')"])
    run([sys.executable, "-m", "unittest", "discover", "-s", "code/tests", "-v"])
    for script in ("floc_msm_main_corrected.py", "cwru_loss_compare_sourceclosed.py", "noisex92_ucr_seeded.py"):
        run([sys.executable, str(CODE / script), "--help"])
    with tempfile.TemporaryDirectory(prefix="floc_msm_smoke_") as temp:
        run([sys.executable, str(CODE / "floc_msm_main_corrected.py"), "--smoke", "--log-dir", str(Path(temp) / "smoke")])
    print("Smoke tests passed. No external datasets were used.")


if __name__ == "__main__":
    main()
