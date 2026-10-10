"""Controlled full-protocol L1 vs FLOC(p=1) reproducibility check."""
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).with_name("floc_msm_main_corrected.py")
OUT = ROOT / "results" / "l1_floc1_full_reproduction"
OUT.mkdir(parents=True, exist_ok=True)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.argv = [
    str(SCRIPT),
    "--losses", "l1,floc_1.0",
    "--n-seeds", "5",
    "--seed", "20261007",
    "--log-dir", str(OUT),
]
runpy.run_path(str(SCRIPT), run_name="__main__")
