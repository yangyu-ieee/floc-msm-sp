"""Run the frozen multi-alpha sweep with the verified symmetric CMS sampler.

The frozen source is never modified. This wrapper applies two auditable text
substitutions in memory: replace its legacy transform in both data generators,
and restrict the sweep to the two impulsive conditions used in the paper.
All other model, loss, masking, and evaluation code remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from stable_noise import from_cms_variates


REVISION_DIR = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = (
    REVISION_DIR
    / "baseline_submission_candidate_2026-06-30"
    / "code"
    / "floc_tsp_loss_compare_fixed.py"
)
OUTPUT_DIR = REVISION_DIR / "logs" / "r6_corrected_multialpha_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

source_bytes = FROZEN_SOURCE.read_bytes()
source = source_bytes.decode("utf-8")
legacy_expression = (
    "np.sin(alpha*U)*(np.cos(U)/W)**((1-alpha)/alpha)"
    "/(np.cos(U)**(1/alpha))"
)
if source.count(legacy_expression) != 2:
    raise RuntimeError(
        "Expected exactly two legacy sampler expressions in the frozen source; "
        "refusing to run an unverified transformation."
    )
source = source.replace(legacy_expression, "from_cms_variates(alpha,U,W)")
alpha_loop_variants = (
    "for alpha in [2.0, 1.5, 1.0]:",
    "for alpha in [2.0,1.5,1.0]:",
)
loop_count = sum(source.count(loop) for loop in alpha_loop_variants)
if loop_count < 3:
    raise RuntimeError("Expected the original sweep and summary alpha loops.")
for loop in alpha_loop_variants:
    source = source.replace(loop, "for alpha in [1.5,1.0]:")
source = source.replace(
    "import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, time, os, json\n",
    "import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, time, os, json\n"
    "from stable_noise import from_cms_variates\n",
    1,
)
if source.count("from stable_noise import from_cms_variates") != 1:
    raise RuntimeError("The CMS helper import was not inserted exactly once.")

seed = 20261008
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

os.environ["FLOC_MSM_SAVE"] = str(OUTPUT_DIR)
start = time.time()
namespace = {"__name__": "__main__", "__file__": str(FROZEN_SOURCE)}
exec(compile(source, str(FROZEN_SOURCE), "exec"), namespace)

metadata = {
    "experiment": "corrected multi-alpha loss sweep",
    "alphas": [1.5, 1.0],
    "losses": namespace["LOSSES"],
    "label_budgets": namespace["N_LABELED"],
    "downstream_seeds": namespace["N_SEEDS"],
    "pretraining_seed": seed,
    "noise_generator": "verified symmetric CMS S1, exp(-|t|^alpha)",
    "normalization": "per-sequence max-absolute after corruption",
    "frozen_source": str(FROZEN_SOURCE),
    "frozen_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "transformed_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "stable_noise_sha256": hashlib.sha256(
        (REVISION_DIR / "code" / "stable_noise.py").read_bytes()
    ).hexdigest(),
    "elapsed_seconds": round(time.time() - start, 3),
}
(OUTPUT_DIR / "run_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
