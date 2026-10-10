"""One-off deterministic reconciliation run for the manuscript's core table."""
from pathlib import Path
import runpy
import sys

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

root = Path(__file__).resolve().parents[1]
runner = Path(__file__).with_name("floc_msm_main_corrected.py")
out = root / "results" / "main_table_reconciliation_20261009"
out.mkdir(parents=True, exist_ok=True)
sys.argv = [
    str(runner),
    "--losses", "mse,l1,huber,floc_1.2",
    "--n-seeds", "5",
    "--seed", "20261007",
    "--log-dir", str(out),
]
runpy.run_path(str(runner), run_name="__main__")
