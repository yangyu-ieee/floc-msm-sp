"""Verify and optionally rerun the frozen corrected multi-alpha sweep.

Default invocation is standard-library-only and does NOT run any experiment:
    python revision_audits/verify_and_rerun_multialpha.py

Optional explicit GPU rerun, writing to an entirely new non-frozen folder:
    python revision_audits/verify_and_rerun_multialpha.py --execute \
        --output-dir /path/to/new-multialpha-rerun

The historical v70 wrapper points to a source path absent from the released
tree. The root-level historical source has exactly the recorded frozen SHA256.
This utility verifies original, corrected in-memory code, wrapper, and stable
noise generator SHA256 values before ANY possible execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "revision_v70_20261010"
FROZEN_SOURCE = ROOT / "floc_tsp_loss_compare_fixed.py"
HISTORICAL_WRAPPER = ARCHIVE / "code/provenance/run_corrected_multialpha_sweep.py"
STABLE = ARCHIVE / "code/stable_noise.py"
METADATA = ARCHIVE / "results/multialpha/corrected_sweep/run_metadata.json"

LEGACY_EXPRESSION = (
    "np.sin(alpha*U)*(np.cos(U)/W)**((1-alpha)/alpha)"
    "/(np.cos(U)**(1/alpha))"
)
ORIGINAL_IMPORT = (
    "import torch, torch.nn as nn, torch.nn.functional as F, "
    "numpy as np, math, time, os, json\n"
)
NEW_IMPORT = ORIGINAL_IMPORT + "from stable_noise import from_cms_variates\n"
ALPHA_LOOPS = (
    "for alpha in [2.0, 1.5, 1.0]:",
    "for alpha in [2.0,1.5,1.0]:",
)


def sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def verify() -> tuple[str, dict]:
    recorded = json.loads(METADATA.read_text(encoding="utf-8"))
    required = {
        str(FROZEN_SOURCE): recorded["frozen_source_sha256"],
        str(HISTORICAL_WRAPPER): recorded["runner_sha256"],
        str(STABLE): recorded["stable_noise_sha256"],
    }
    for name, expected in required.items():
        actual = sha256(Path(name).read_bytes())
        if actual != expected:
            raise RuntimeError(f"Hash mismatch: {name}: {actual} != {expected}")
        print(f"PASS source SHA256: {name}")
    source = FROZEN_SOURCE.read_text(encoding="utf-8")
    if source.count(LEGACY_EXPRESSION) != 2:
        raise RuntimeError("Original sampler expression does not occur exactly twice.")
    if sum(source.count(v) for v in ALPHA_LOOPS) != 3:
        raise RuntimeError("Expected three multi-alpha loops, including summary loops.")
    if source.count(ORIGINAL_IMPORT) != 1:
        raise RuntimeError("Historical import declaration does not match.")
    corrected = source.replace(LEGACY_EXPRESSION, "from_cms_variates(alpha,U,W)")
    for original in ALPHA_LOOPS:
        corrected = corrected.replace(original, "for alpha in [1.5,1.0]:")
    corrected = corrected.replace(ORIGINAL_IMPORT, NEW_IMPORT, 1)
    if sha256(corrected.encode("utf-8")) != recorded["transformed_source_sha256"]:
        raise RuntimeError("Reconstructed corrected in-memory source hash mismatch.")
    # Confirm that both classification train and test generators receive alpha.
    if ("Xtr,ytr=gen_cls(n_train,128,alpha); Xte,yte=gen_cls(n_test,128,alpha)"
            not in corrected):
        raise RuntimeError("Expected downstream alpha flow is absent.")
    if "X_pt=gen_unlabeled(10000,128,alpha)" not in corrected:
        raise RuntimeError("Expected pretraining alpha flow is absent.")
    if "torch.manual_seed(seed); np.random.seed(seed)" not in corrected:
        raise RuntimeError("Downstream seed reset no longer matches archive.")
    if corrected.count("for alpha in [1.5,1.0]:") != 3:
        raise RuntimeError("Corrected alpha loops no longer match archive.")
    if recorded["alphas"] != [1.5, 1.0] or recorded["pretraining_seed"] != 20261008:
        raise RuntimeError("Unexpected archived sweep arguments.")
    print("PASS corrected transformed-source SHA256:", recorded["transformed_source_sha256"])
    print("PASS alpha=1.5 and alpha=1.0 propagate into pretraining AND downstream.")
    print("NOTE: loss-wise pretraining initialization/shuffle are NOT independently")
    print("      reseeded to a common stream; only downstream seeds are reset.")
    print("Verification complete. No model training was performed.")
    return corrected, recorded



def audit_historical_alpha2_branch() -> None:
    """The historical 'Gaussian alpha=2' label is actually jitter-only."""
    source = FROZEN_SOURCE.read_text(encoding="utf-8")
    helper = STABLE.read_text(encoding="utf-8")
    historical = json.loads(
        (ROOT / "tsp_loss_compare_summary.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    # Both the unlabeled and the labeled/test generators skip the nominal
    # stable noise injection entirely at alpha=2; only Gaussian jitter is left.
    assert source.count("if alpha<2.0:") == 2
    assert "sig+=0.02*np.random.randn(seq_len)" in source
    assert "sig+=0.05*np.random.randn(seq_len)" in source
    assert "sig+=noise.astype(np.float32)*0.15" in source
    assert "sig+=noise.astype(np.float32)*0.3" in source
    assert source.count("for alpha in [2.0,") == 3
    # The mathematically correct S_alpha S alpha=2 convention is Gaussian
    # with sigma=sqrt(2)*scale, NOT sigma=0.02 or sigma=0.05.
    assert "np.sqrt(2.0) * scale" in helper
    assert "alpha=2 requires Gaussian variates" in helper
    assert metadata["alphas"] == [1.5, 1.0]
    old = historical["best_by_alpha"]["2.0"]
    assert set(old) == {"50", "100", "200", "500"}
    for item in old.values():
        assert item["best_mean"] == item["second_mean"] == 1.0
    print("PASS historical alpha=2 source inspection: the two generators")
    print("      skip the nominal alpha-stable/Gaussian noise branch at alpha=2.")
    print("      They retain jitter SD=0.02 (pretrain), SD=0.05 (downstream).")
    print("      Historical 100% top-two means are a JITTER-ONLY control,")
    print("      NOT an alpha=2 Gaussian-injection result at scales 0.15/0.30.")
    print("PASS the corrected archival multi-alpha sweep excludes alpha=2.")


def audit_empirical_rank_margins() -> None:
    base = ARCHIVE / "results/multialpha/corrected_sweep"
    data = json.loads((base / "results.json").read_text(encoding="utf-8"))
    phase = json.loads((base / "phase_map_summary.json").read_text(encoding="utf-8"))
    close = []
    print("Alpha, labeled N, top loss, second loss, gap in correct predictions / 2500")
    for alpha in ("1.5", "1.0"):
        for n in ("50", "100", "200", "500"):
            key = f"alpha={alpha}/n={n}"
            record = phase["cells"][key]
            def ncorrect(loss: str) -> int:
                vals = data[f"a={alpha}/loss={loss}/n={n}"]["seeds"]
                assert len(vals) == 5
                counts = [round(500 * val) for val in vals]
                assert all(abs(c/500 - val) < 1e-7
                           for c, val in zip(counts, vals))
                return sum(counts)
            first = ncorrect(record["best_loss"])
            second = ncorrect(record["second_loss"])
            gap = first - second
            assert gap >= 0
            assert abs(gap/2500 - record["best_second_margin"]) < 1e-6
            print(f"{alpha} {n}: {record['best_loss']} vs "
                  f"{record['second_loss']}, gap {gap}/2500")
            if gap <= 2:
                close.append((alpha, n))
    assert close == [("1.5", "100"), ("1.5", "500")]
    print("PASS: top-loss rankings have two near-ties (2/2500 each).")

    # The targeted alpha=1.0 runner is exact Cauchy at alpha=1,
    # despite retaining the historically incorrect general-alpha transform.
    target = (ARCHIVE / "code/alpha1_training_dynamics"
              / "targeted_alpha10_fullseed.py").read_text(encoding="utf-8")
    cauchy_cfg = json.loads(
        (ARCHIVE / "results/cauchy_level/config.json").read_text(encoding="utf-8")
    )
    assert cauchy_cfg["alpha"] == 1.0 and cauchy_cfg["seeds"] == 10
    assert 'parser.add_argument("--alpha", type=float, default=1.0)' in target
    assert "np.sin(alpha * u)" in target and "np.cos(u) / w" in target
    print("PASS: archived targeted run is alpha=1, where the legacy formula")
    print("      algebraically equals tan(U). Do NOT use that target script at alpha!=1.")



def rerun(corrected: str, recorded: dict, output_dir: Path) -> None:
    destination = output_dir.expanduser().resolve()
    if ARCHIVE.resolve() == destination or ARCHIVE.resolve() in destination.parents:
        raise ValueError("Refusing to write rerun artifacts inside frozen v70.")
    if destination.exists():
        raise FileExistsError(f"Output directory must not yet exist: {destination}")
    # Import heavy packages only with explicit --execute.
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Historical source requires CUDA. GPU rerun cancelled.")
    destination.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str((ARCHIVE / "code").resolve()))
    os.environ["FLOC_MSM_SAVE"] = str(destination)
    seed = recorded["pretraining_seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The original source itself is immutable: evaluate only the verified
    # in-memory corrected version and write outputs to a fresh directory.
    compiled = compile(corrected, str(FROZEN_SOURCE), "exec")
    namespace = {"__name__": "__main__", "__file__": str(FROZEN_SOURCE)}
    exec(compiled, namespace)
    evidence = {
        "kind": "newly executed rerun from SHA-verified source, not original v70 run",
        "frozen_source_sha256": recorded["frozen_source_sha256"],
        "transformed_source_sha256": recorded["transformed_source_sha256"],
        "stable_noise_sha256": recorded["stable_noise_sha256"],
        "original_runner_sha256": recorded["runner_sha256"],
        "pretraining_seed": seed,
        "device": str(torch.cuda.get_device_name(0)),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "notice": "Cross-hardware bitwise reproducibility is not implied.",
    }
    (destination / "rerun_provenance.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly run a NEW, potentially long CUDA experiment")
    parser.add_argument("--output-dir", type=Path,
                        help="Mandatory fresh folder outside frozen v70 for --execute")
    args = parser.parse_args()
    corrected, recorded = verify()
    audit_empirical_rank_margins()
    audit_historical_alpha2_branch()
    if args.execute:
        if args.output_dir is None:
            parser.error("--execute requires --output-dir; frozen results never overwritten")
        rerun(corrected, recorded, args.output_dir)
    elif args.output_dir is not None:
        parser.error("--output-dir is only meaningful with --execute")


if __name__ == "__main__":
    main()
