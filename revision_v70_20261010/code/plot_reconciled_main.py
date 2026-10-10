"""Rebuild main/Supplement main-benchmark figures from the reconciled run."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
summary = json.loads(
    (root / "results" / "main_table_reconciliation_20261009" / "summary.json")
    .read_text(encoding="utf-8")
)
losses = [
    ("mse", "MSE", "#0072B2"),
    ("l1", "L1", "#D55E00"),
    ("huber", "Huber", "#009E73"),
    ("floc_1.2", r"$p$-moment ($p=1.2$)", "#CC79A7"),
]
budgets = (50, 100, 200, 500)
fig, ax = plt.subplots(figsize=(5.3, 3.3), constrained_layout=True)
for key, label, color in losses:
    means = [100 * summary[key][str(n)]["mean"] for n in budgets]
    stds = [100 * summary[key][str(n)]["std"] for n in budgets]
    ax.errorbar(
        budgets, means, yerr=stds, marker="o", linewidth=1.7,
        capsize=3, label=label, color=color,
    )
ax.set_xlabel("Labeled examples")
ax.set_ylabel("Test accuracy (%)")
ax.set_xticks(budgets)
ax.set_ylim(85, 101)
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, ncol=2, fontsize=8)
for rel in (Path("manuscript/figures/sp_main_alpha15.pdf"),
            Path("supplement/figures/sp_main_alpha15.pdf")):
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    print(target)
plt.close(fig)
