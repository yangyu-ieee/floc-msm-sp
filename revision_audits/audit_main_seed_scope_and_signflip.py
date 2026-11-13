"""26SP R1: inspect synthetic main data-generation scope and exact five-seed tests.

No retraining; uses Python standard library only.
Run from repository root:
    python revision_audits/audit_main_seed_scope_and_signflip.py

The AST/source assertions are deliberately specific to the frozen v70 runner.
They detect a future protocol change; they do not produce new empirical results.
"""
from __future__ import annotations

import ast
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


if __name__ == "__main__":
    source_evidence()
    numerical_evidence()
