"""Cauchy (alpha=1) ten-pipeline evidence, Holm multiplicity and legacy-summary audit.

Usage (from repo root, with scipy installed):
    python revision_audits/audit_cauchy_welch_holm_and_legacy_summary.py

Reads immutable frozen v70 source and archived per-seed accuracies; never
re-trains, edits old results, downloads data, or treats distinct loss-specific
random seeds as matched pairs.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_ind

ROOT = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
CODE = ROOT / "code/alpha1_training_dynamics/targeted_alpha10_fullseed.py"
LEGACY_CODE = ROOT / "code/audit_cauchy_unpaired_welch.py"
BASE = ROOT / "results/cauchy_level"
BUDGETS = (50, 100, 200, 500)
RAW_REFERENCE = (1.09e-5, 4.41e-7, 2.62e-8, 2.47e-5)
HOLM_REFERENCE = (2.18e-5, 1.32e-6, 1.05e-7, 2.47e-5)
GAIN_PP = (14.92, 12.64, 13.84, 5.72)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def holm_adjust(raw):
    indices = sorted(range(len(raw)), key=lambda i: raw[i])
    adjusted = [0.0] * len(raw)
    running_max = 0.0
    for rank, ix in enumerate(indices):
        running_max = max(running_max, (len(raw) - rank) * raw[ix])
        adjusted[ix] = min(1.0, running_max)
    return adjusted


def exact_independent_permutation_count(a, b):
    """Enumerate all C(20,10) assignments under a distribution-exchangeable null.

    Use integer correct counts out of 500 to avoid floating point tie errors.
    This is a distribution-null sensitivity analysis, not an exact Welch test
    of equality of means under arbitrary heterogeneous variances.
    """
    aa = [round(float(x) * 500) for x in a]
    bb = [round(float(x) * 500) for x in b]
    assert len(aa) == len(bb) == 10
    assert all(abs(float(x)-k/500) < 1e-7
               for x,k in zip([*a,*b], [*aa,*bb]))
    pooled = aa + bb
    total = sum(pooled)
    observed = abs(sum(aa) - sum(bb))
    extreme, enumerated = 0, 0
    for indices in combinations(range(20), 10):
        selected = sum(pooled[i] for i in indices)
        enumerated += 1
        if abs(2 * selected - total) >= observed:
            extreme += 1
    assert enumerated == 184756
    return extreme, enumerated


def audit():
    config = read_json(BASE / "config.json")
    results = read_json(BASE / "results.json")
    summary = read_json(BASE / "summary.json")
    source = CODE.read_text(encoding="utf-8")
    original_audit = LEGACY_CODE.read_text(encoding="utf-8")

    assert config["alpha"] == 1.0 and config["seeds"] == 10
    assert config["n_test"] == 500 and config["n_labeled"] == list(BUDGETS)
    assert config["script_sha256"] == hashlib.sha256(CODE.read_bytes()).hexdigest()
    assert config["losses"] == [
        "mse", "l1", "charbonnier", "floc_1.0", "floc_1.2", "floc_1.5"
    ]
    assert "set_seed(seed * 1000 + LOSS_SEED_OFFSET.get(loss, 199))" in source
    assert "seed * 10000 + n * 37 + LOSS_SEED_OFFSET.get(loss, 199)" in source
    assert '"mse": 11' in source and '"floc_1.2": 103' in source
    assert "def paired(" in source and "stats.ttest_rel(aa, bb)" in source
    assert "ttest_ind(" in original_audit and "equal_var=False" in original_audit

    # The alpha=1 legacy expression simplifies to tan(U), even though it is
    # invalid for non-Cauchy alpha. Never generalize the historical runner.
    assert "np.sin(alpha * u)" in source
    assert "(np.cos(u) / w) ** ((1 - alpha) / alpha)" in source
    print("PASS source: alpha=1 targeted run uses independent, loss-offset streams")
    print("     and ten separately pretrained encoders per loss. The historical")
    print("     summary used paired tests, but the R1 correction uses Welch.")

    pvalues = []
    for budget, gain in zip(BUDGETS, GAIN_PP):
        a = np.array(results[f"loss=floc_1.2/n={budget}"]["seeds"],
                     dtype=np.float64)
        b = np.array(results[f"loss=mse/n={budget}"]["seeds"],
                     dtype=np.float64)
        assert a.shape == b.shape == (10,)
        assert all(abs(v - round(500*v)/500) < 1e-7
                   for v in np.concatenate([a,b]))
        assert abs(100 * (a.mean()-b.mean()) - gain) < 1e-4
        stat = ttest_ind(a, b, equal_var=False)
        pvalues.append(float(stat.pvalue))
        var_a, var_b = a.var(ddof=1), b.var(ddof=1)
        se = math.sqrt(var_a/10 + var_b/10)
        df = (var_a/10 + var_b/10)**2 / (
            (var_a/10)**2/9 + (var_b/10)**2/9
        )
        half_width = t.ppf(.975, df) * se * 100
        print(f"n={budget}: mean difference={gain:+.2f} pp; "
              f"Welch p={stat.pvalue:.8g}; "
              f"descriptive unadjusted 95% CI=({gain-half_width:.2f},"
              f"{gain+half_width:.2f}) pp")
        # The frozen legacy paired result must not be cited as the corrected
        # independent-sample p value.
        legacy_p = summary["comparisons"][str(budget)][
            "best_floc_vs_mse"
        ]["p_value"]
        assert legacy_p > stat.pvalue

    corrected = holm_adjust(pvalues)
    for p, ref in zip(pvalues, RAW_REFERENCE):
        assert abs(p-ref) / ref < 0.01
    for p, ref in zip(corrected, HOLM_REFERENCE):
        assert abs(p-ref) / ref < 0.01
    assert all(p < .001 for p in corrected)
    print("Holm-adjusted p in budget order:",
          ", ".join(f"{p:.8g}" for p in corrected))
    print("PASS: four independent-stream Welch p-values remain <0.001")
    print("      after Holm correction, as current submitted R1 claims.")

    for other in ("l1", "charbonnier"):
        vals = []
        for budget in BUDGETS:
            a = results[f"loss=floc_1.2/n={budget}"]["seeds"]
            b = results[f"loss={other}/n={budget}"]["seeds"]
            vals.append(float(ttest_ind(a,b,equal_var=False).pvalue))
        assert all(p > 0.05 for p in vals), (other, vals)
        print(f"Unadjusted Welch p vs {other}:",
              ", ".join(f"{p:.5f}" for p in vals))
    print("Exact independent-group permutation sensitivity (10+10; 184,756 assignments):")
    perm_rows = {}
    selected_raw = []
    all_raw = []
    for budget in BUDGETS:
        aa = results[f"loss=floc_1.2/n={budget}"]["seeds"]
        for baseline in ("mse", "l1", "charbonnier"):
            bb = results[f"loss={baseline}/n={budget}"]["seeds"]
            extreme, total = exact_independent_permutation_count(aa, bb)
            pv = extreme / total
            perm_rows[(budget, baseline)] = (extreme, pv)
            all_raw.append(pv)
            if baseline == "mse":
                selected_raw.append(pv)
            print(f"  n={budget}, p=1.2 vs {baseline}: {extreme}/{total} = {pv:.9f}")
    assert [perm_rows[(budget,"mse")][0] for budget in BUDGETS] == [2,4,2,16]
    assert all(perm_rows[(budget,baseline)][1] > .05
               for budget in BUDGETS
               for baseline in ("l1","charbonnier"))
    selected_holm = holm_adjust(selected_raw)
    assert all(p < .001 for p in selected_holm)
    assert [round(x*184756,8) for x in selected_holm] == [8,8,8,16]
    # Re-index Holm across all 12 comparisons, not just four MSE contrasts.
    all_holm = holm_adjust(all_raw)
    mse_positions = (0, 3, 6, 9)
    mse_12_holm = [all_holm[i] for i in mse_positions]
    assert [round(x*184756,8) for x in mse_12_holm] == [24,40,24,144]
    assert all(p < .001 for p in mse_12_holm)
    print("4-MSE-family Holm (50/100/200/500):",
          ", ".join(f"{p:.9f}" for p in selected_holm))
    print("12-contrast-family Holm vs MSE:",
          ", ".join(f"{p:.9f}" for p in mse_12_holm))
    print("PASS: four MSE gains remain below 0.001 under either illustrative")
    print("      multiplicity family; neither L1 nor Charbonnier is significant.")
    print("NOTE: ordinary permutation test assumes equal group distributions")
    print("      under H0; this is NOT a distribution-free Welch p-value, and")
    print("      exploratory prior model/hypothesis selection is not corrected.")

    print("CAUTION: comparing loss-specific streams does not hold test")
    print("         realizations fixed. The 40 budget-by-seed observations")
    print("         reuse 10 pretraining runs and are not independent.")
    print("PASS: archive only; no frozen files modified.")


if __name__ == "__main__":
    audit()
