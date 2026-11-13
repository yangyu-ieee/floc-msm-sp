"""26SP R1: inspect synthetic main data-generation scope and exact five-seed tests.

No retraining; uses Python standard library only.
Run from repository root:
    python revision_audits/audit_main_seed_scope_and_signflip.py

The AST/source assertions are deliberately specific to the frozen v70 runner.
They detect a future protocol change; they do not produce new empirical results.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
from math import isclose
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
SRC = BASE / "code/floc_msm_main_corrected.py"
RESULTS = BASE / "results/main_table_reconciliation_20261009/results.json"
BUDGETS = ("50", "100", "200", "500")


def exact_paired_signflip(a: list[float], b: list[float]) -> float:
    """Enumerated two-sided sign-flip statistic; formal exactness assumes sign symmetry."""
    diffs = [x - y for x, y in zip(a, b)]
    assert len(diffs) == len(a) == len(b) == 5
    observed = abs(sum(diffs))
    extreme = sum(
        abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(diffs))
    )
    return extreme / (2 ** len(diffs))


def source_evidence() -> None:
    source = SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = {x.name: x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "eval_downstream" in funcs and "gen_cls" in funcs

    call_targets = [
        x.func.id for x in ast.walk(funcs["eval_downstream"])
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
    ]
    # Important order: gen_cls(n_train) consumes NumPy RNG *before* gen_cls(n_test).
    assert call_targets.count("gen_cls") == 2
    assert source.index("Xtr,ytr=gen_cls(n_train") < source.index("Xte,yte=gen_cls(n_test")
    # Current source seeds both RNGs immediately before each downstream evaluation.
    assert source.count("torch.manual_seed(seed); np.random.seed(seed)") >= 2
    assert "X_pt=gen_unlabeled(CONFIG['pt_samples'],128,CONFIG['alpha'])" in source
    assert "for loss_type in CONFIG['losses']:" in source
    assert "for n_labeled in CONFIG['n_labeled']:" in source
    assert "cls=i%5" in source
    assert "for i in range(n):" in source
    assert "sig+=0.05*np.random.randn(seq_len)" in source
    assert "X/=(np.abs(X).max(axis=1,keepdims=True)+1e-8)" in source
    assert "CONFIG['ft_epochs'] if n_labeled<=200 else max(CONFIG['ft_epochs'],50)" in source
    print("PASS: each budget/seed resets RNGs, then generates training followed by test data.")
    print("  Within a budget/seed, different losses have common synthetic data draws.")
    print("  Across budgets, the SAME seed creates prefix-nested training draws,")
    print("  but n_train changes RNG consumption before the 500-sample test draw.")
    print("  500 labels receives 50 fine-tuning epochs; <=200 labels receive 30.")
    print("  Five-seed SD includes test noise/data realization plus training and fine-tuning")
    print("  variability, but not independent pretraining checkpoints.")


def numerical_evidence() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    expected_mse_p = (0.5625, 0.1875, 0.1875, 0.1875)
    expected_l1_p = (0.625, 0.1875, 0.8125, 0.875)
    for index, n in enumerate(BUDGETS):
        p12 = data["floc_1.2"][n]["seeds"]
        mse = data["mse"][n]["seeds"]
        l1 = data["l1"][n]["seeds"]
        p_mse = exact_paired_signflip(p12, mse)
        p_l1 = exact_paired_signflip(p12, l1)
        assert isclose(p_mse, expected_mse_p[index], abs_tol=1e-12), (n,p_mse)
        assert isclose(p_l1, expected_l1_p[index], abs_tol=1e-12), (n,p_l1)
        mean_pp_mse = (sum(p12)-sum(mse)) / 5 * 100
        print(f"n={n:>3}: p1.2–MSE={mean_pp_mse:+.3f}pp, exact p={p_mse:.4f}; "
              f"p1.2–L1 exact p={p_l1:.4f}")
    print("PASS: exact finite-sample sign-flip enumeration from frozen v70 per-seed data.")


def audit_two_archived_main_runs() -> None:
    """Different frozen run outputs are not the same 'Table 1' evidence.

    This source/JSON-only check verifies explicit supersession, common
    runner hash, pretrain trajectory differences, and the historical
    provenance wrapper's broken *current archived path* after relocation.
    It cannot determine why the archived GPU runs differ.
    """
    parent = BASE / "results"
    canonical = json.loads((parent /
        "main_table_reconciliation_20261009/results.json").read_text(encoding="utf-8"))
    earlier = json.loads((parent / "main_alpha15/results.json").read_text(encoding="utf-8"))
    canonical_cfg = json.loads((parent /
        "main_table_reconciliation_20261009/config.json").read_text(encoding="utf-8"))
    earlier_cfg = json.loads((parent /
        "main_alpha15/config.json").read_text(encoding="utf-8"))
    canonical_meta = json.loads((parent /
        "main_table_reconciliation_20261009/run_metadata.json").read_text(encoding="utf-8"))
    earlier_meta = json.loads((parent /
        "main_alpha15/run_metadata.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    assert sha == canonical_cfg["code_sha256"] == earlier_cfg["code_sha256"]
    assert sha == canonical_meta["runner_script_sha256"] == earlier_meta["script_sha256"]
    assert canonical_meta["deterministic_algorithms"] is True
    assert earlier_meta["deterministic_algorithms"] is True
    assert "superseded for manuscript Table 1" in canonical_meta["notes"]
    assert canonical_cfg["seed"] == earlier_cfg["seed"] == 20261007
    assert canonical_cfg["n_labeled"] == earlier_cfg["n_labeled"]
    assert canonical_cfg["mask_strategy"] == earlier_cfg["mask_strategy"] == "random_token"
    assert canonical_cfg["mask_count_rule"] == earlier_cfg["mask_count_rule"]
    assert canonical_cfg["losses"] == ["mse", "l1", "huber", "floc_1.2"]
    assert len(earlier_cfg["losses"]) > len(canonical_cfg["losses"])

    # The wrapper archived *under code/provenance/* used a sibling
    # lookup that would have worked while the script lived under code/.
    # But at its CURRENT archived path the sibling is absent.
    moved_wrapper = BASE / "code/provenance/run_main_table_reconciliation_executed_20261009.py"
    wrapper_src = moved_wrapper.read_text(encoding="utf-8")
    assert 'runner = Path(__file__).with_name("floc_msm_main_corrected.py")' in wrapper_src
    wrong_path = moved_wrapper.with_name("floc_msm_main_corrected.py")
    assert not wrong_path.is_file()
    assert SRC.is_file()
    # The canonical portable entrypoint has the correct sibling location:
    runnable = BASE / "code/run_main_table_reconciliation.py"
    runnable_src = runnable.read_text(encoding="utf-8")
    assert 'runner = Path(__file__).with_name("floc_msm_main_corrected.py")' in runnable_src
    assert runnable.with_name("floc_msm_main_corrected.py") == SRC
    assert canonical_meta["wrapper_script"].startswith("code/provenance/")

    expected_new_minus_old_pp = {
        "mse": [-.64, -.44, -.12, -.28],
        "l1": [+.64, +.52, -.08, +.32],
        "huber": [-1.24, -.92, -.08, -.64],
        "floc_1.2": [-.32, +.12, +.28, +.36],
    }
    for loss, shifts in expected_new_minus_old_pp.items():
        for n, expected_shift in zip(BUDGETS, shifts):
            old = earlier[loss][n]["seeds"]
            new = canonical[loss][n]["seeds"]
            assert len(old) == len(new) == 5
            old_count = sum(round(v*500) for v in old)
            new_count = sum(round(v*500) for v in new)
            assert isclose(100*(new_count-old_count)/2500,
                           expected_shift, abs_tol=1e-8)
            assert old_count != new_count
            print(f"{loss:>10}, n={n}: previous={old_count/25:.2f}%, "
                  f"Table1={new_count/25:.2f}%, shift={expected_shift:+.2f}pp")

    # A stronger cross-run discrepancy is visible in SCRATCH, which
    # initializes from None and never loads a pretrained encoder.
    # Therefore loss-list order/pretraining trajectory cannot be the
    # *sole* mechanism behind the two-run differences.
    source_text = SRC.read_text(encoding="utf-8")
    assert earlier_cfg["losses"][:2] == ["scratch", "mse"]
    assert canonical_cfg["losses"][0] == "mse"
    assert "if loss_type=='scratch': continue" in source_text
    assert "torch.manual_seed(args.seed + 1)" in source_text
    assert "generator=torch.Generator().manual_seed(args.seed + 2)" in source_text
    assert "a=eval_downstream(None,n_labeled" in source_text
    assert source_text.count("torch.manual_seed(seed); np.random.seed(seed)") >= 2
    # In both archives MSE is the first trained loss; the earlier
    # apparent "scratch" preceding MSE is skipped in the pretrain loop.
    # Consequently MSE pretraining trajectory differences are not
    # explained by loss objectives trained before MSE.
    scratch_shifts = (+.24, +3.72, -.44, +.28)
    for n, expected_shift in zip(BUDGETS, scratch_shifts):
        old_s = earlier["scratch"][n]["seeds"]
        new_s = canonical["scratch"][n]["seeds"]
        assert len(old_s) == len(new_s) == 5
        old_counts = [round(v*500) for v in old_s]
        new_counts = [round(v*500) for v in new_s]
        shift = (sum(new_counts)-sum(old_counts))/25
        assert isclose(shift, expected_shift, abs_tol=1e-10)
        assert old_counts != new_counts
        print(f"SCRATCH, n={n}: prior={sum(old_counts)/25:.2f}%, "
              f"Table1={sum(new_counts)/25:.2f}%, shift={shift:+.2f}pp")
        if n == "100":
            assert old_counts[2] == 175 and new_counts[2] == 242
            assert new_counts[2]-old_counts[2] == 67
    assert canonical_meta["cudnn_deterministic"] is True
    assert canonical_meta["cudnn_benchmark"] is False
    assert "cudnn_deterministic" not in earlier_meta
    assert "cudnn_benchmark" not in earlier_meta
    print("PASS: SCRATCH differs at every budget despite no pretrained weights.")
    print("      Earliest trained objective MSE is first in both archives.")
    print("      Backend flag coverage is incomplete in earlier metadata.")
    print("      Neither the GPU cause nor bitwise input identity is proven.")

    for loss in ("mse", "floc_1.2"):
        old_curve = json.loads((parent /
            f"main_alpha15/pretraining_trajectory_{loss}.json").read_text(encoding="utf-8"))
        new_curve = json.loads((parent /
            f"main_table_reconciliation_20261009/pretraining_trajectory_{loss}.json").read_text(encoding="utf-8"))
        assert len(old_curve) == len(new_curve) == 20
        assert not isclose(old_curve[-1]["mean_train_loss"],
                           new_curve[-1]["mean_train_loss"],
                           abs_tol=1e-12)
    for n in BUDGETS:
        old_diff = (earlier["floc_1.2"][n]["mean"] - earlier["mse"][n]["mean"])*100
        new_diff = (canonical["floc_1.2"][n]["mean"] - canonical["mse"][n]["mean"])*100
        assert old_diff > 0 and new_diff > 0
    print("PASS: shared source SHA and seeds but 16 non-identical run cells;")
    print("      Table 1 must cite the explicitly superseding reconciliation run.")
    print("      Across runs, p=1.2 exceeds MSE in mean at all four budgets")
    print("      but the contrast magnitude is not the same across the archives.")
    print("SOURCE-PATH CAVEAT: the moved historical executed-wrapper fails at")
    print("      code/provenance/; use code/run_main_table_reconciliation.py")
    print("      from an authorized clone with explicit separate output folder.")
    print("No source modification or GPU rerun performed.")




if __name__ == "__main__":
    source_evidence()
    numerical_evidence()
    audit_two_archived_main_runs()
