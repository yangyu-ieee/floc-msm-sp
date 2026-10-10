"""Reproduce the deterministic five-loss alpha=1.5 main-table comparison.

By default, outputs go to a timestamped reproduction directory so archived
submission results are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="")
args = parser.parse_args()

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

root = Path(__file__).resolve().parents[1]
runner = Path(__file__).with_name("floc_msm_main_corrected.py")
if args.output_dir:
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
else:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = root / "results" / "reproduction" / f"main_table_reconciliation_{stamp}"
out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
sys.argv = [
    str(runner),
    "--losses", "mse,l1,huber,floc_1.2",
    "--n-seeds", "5",
    "--seed", "20261007",
    "--log-dir", str(out),
]
runpy.run_path(str(runner), run_name="__main__")
metadata = {
    "experiment": "deterministic alpha=1.5 main-table reconciliation reproduction",
    "elapsed_seconds": time.perf_counter() - started,
    "python": sys.version,
    "torch": torch.__version__,
    "numpy": __import__("numpy").__version__,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "seed": 20261007,
    "master_seeds": [0, 1, 2, 3, 4],
    "runner_script_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    "wrapper_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "archived_primary_outputs_are_not_overwritten": True,
}
(out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
