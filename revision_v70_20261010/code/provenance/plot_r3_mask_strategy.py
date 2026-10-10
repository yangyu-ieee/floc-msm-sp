"""Plot normalized pretraining objectives for random-token vs block masking."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
losses = ("mse", "l1", "charbonnier", "floc_1.2")
labels = {"mse": "MSE", "l1": "L1", "charbonnier": "Charbonnier", "floc_1.2": r"$p$-moment loss ($p=1.2$)"}
colors = {"random_token": "#0072B2", "contiguous_block": "#D55E00"}
names = {"random_token": "Random tokens", "contiguous_block": "Contiguous block"}
directories = {"random_token": "r3_mask_strategy_random_token_v2", "contiguous_block": "r3_mask_strategy_contiguous_block_v2"}
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, constrained_layout=True)
for ax, loss in zip(axes.flat, losses):
    for strategy in colors:
        path = ROOT / "logs" / directories[strategy] / f"pretraining_trajectory_{loss}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        values = np.asarray([row["mean_train_loss"] for row in rows], dtype=float)
        ax.plot(np.arange(1, len(values) + 1), values / values[0], marker="o", markersize=2.5,
                linewidth=1.4, color=colors[strategy], label=names[strategy])
    ax.set_title(labels[loss], fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
for ax in axes[:, 0]:
    ax.set_ylabel("Training objective / epoch-1 value")
axes[1, 0].set_xlabel("Pretraining epoch")
axes[1, 1].set_xlabel("Pretraining epoch")
axes[0, 0].legend(frameon=False, fontsize=8)
out = ROOT / "manuscript" / "figures" / "r3_mask_strategy_training_curves.pdf"
fig.savefig(out, bbox_inches="tight")
print(out)
