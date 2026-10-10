"""Recompute the four planned independent-sample Cauchy-level comparisons."""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "results/cauchy_level/results.json").read_text(encoding="utf-8"))
BUDGETS = ("50", "100", "200", "500")

raw = [
    float(ttest_ind(
        RESULTS[f"loss=floc_1.2/n={budget}"]["seeds"],
        RESULTS[f"loss=mse/n={budget}"]["seeds"],
        equal_var=False,
    ).pvalue)
    for budget in BUDGETS
]
order = sorted(range(len(raw)), key=raw.__getitem__)
adjusted = [0.0] * len(raw)
running_max = 0.0
for rank, index in enumerate(order):
    running_max = max(running_max, (len(raw) - rank) * raw[index])
    adjusted[index] = min(1.0, running_max)

expected_raw = [1.09e-5, 4.41e-7, 2.62e-8, 2.47e-5]
expected_holm = [2.18e-5, 1.32e-6, 1.05e-7, 2.47e-5]
for actual, expected in zip(raw, expected_raw):
    assert abs(actual - expected) <= expected * 0.01, (actual, expected)
for actual, expected in zip(adjusted, expected_holm):
    assert abs(actual - expected) <= expected * 0.01, (actual, expected)

print("Independent Welch two-sided p-values, budgets 50/100/200/500:")
print("raw:  " + ", ".join(f"{value:.3g}" for value in raw))
print("Holm: " + ", ".join(f"{value:.3g}" for value in adjusted))
print("PASS: archived runs are analyzed as independent streams, not paired.")
