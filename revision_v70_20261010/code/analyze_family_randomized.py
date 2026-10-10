"""Summarize the five matched randomized-family runs without cross-task tests."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "family_randomized"
SEEDS = range(5)
BUDGETS = (50, 100, 200, 500)
LOSSES = ("scratch", "mse", "l1", "huber", "charbonnier", "floc_1.2")
rows = []
raw = {loss: {n: [] for n in BUDGETS} for loss in LOSSES}
for seed in SEEDS:
    path = RUNS / f"seed_{seed}" / "results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    for loss in LOSSES:
        for n in BUDGETS:
            values = result[loss][str(n)]["seeds"]
            if len(values) != 1:
                raise ValueError(f"Expected one run per invocation: {path} {loss}/{n}")
            raw[loss][n].append(float(values[0]))

def exact_signflip_p(diffs):
    d = np.asarray(diffs, dtype=float)
    observed = abs(d.mean())
    stats = [abs(np.mean(d * np.asarray(signs))) for signs in itertools.product((-1, 1), repeat=len(d))]
    return sum(x >= observed - 1e-15 for x in stats) / len(stats)

summary = {"task": "family_randomized", "alpha": 1.5, "seeds": 5, "pretraining_per_loss_seed": 1, "fine_tune_seeds_per_checkpoint": 1, "losses": {}}
for loss in LOSSES:
    summary["losses"][loss] = {}
    for n in BUDGETS:
        x = np.asarray(raw[loss][n])
        entry = {"seed_accuracies": x.tolist(), "mean": float(x.mean()), "sd_population": float(x.std(ddof=0))}
        if loss != "mse":
            d = x - np.asarray(raw["mse"][n])
            entry["delta_vs_mse_pp"] = float(d.mean() * 100)
            entry["paired_exact_signflip_p_vs_mse"] = exact_signflip_p(d)
        summary["losses"][loss][str(n)] = entry
        rows.append((loss, n, *x.tolist(), float(x.mean()), float(x.std(ddof=0))))

(ROOT / "results" / "family_randomized_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
csv_lines = ["loss,budget,seed0,seed1,seed2,seed3,seed4,mean,sd_population"]
csv_lines += [f"{loss},{n}," + ",".join(f"{v:.6f}" for v in (*vals, mean, sd)) for loss, n, *vals, mean, sd in rows]
(ROOT / "results" / "family_randomized_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

md = ["# Randomized-family synthetic diagnostic", "", "All values are test accuracy (%), mean ± population SD across five matched complete seeds. Each loss was pretrained once per seed; each seed/loss/budget used one matched downstream fine-tuning run. Paired comparisons are within this randomized task only. The manuscript calls the proposed objective the $p$-moment loss; `floc_1.2` remains its legacy result key.", "", "| Labels | Scratch | MSE | L1 | Huber | Charbonnier | $p$-moment loss ($p=1.2$) | $p$-moment loss−MSE (pp), exact p |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
for n in BUDGETS:
    cells = []
    for loss in LOSSES:
        x = np.asarray(raw[loss][n])
        cells.append(f"{100*x.mean():.2f} ± {100*x.std(ddof=0):.2f}")
    d = np.asarray(raw["floc_1.2"][n]) - np.asarray(raw["mse"][n])
    p = exact_signflip_p(d)
    cells.append(f"{d.mean()*100:+.2f}; p={p:.4f}")
    md.append("| " + str(n) + " | " + " | ".join(cells) + " |")
md += ["", "## Seed-level paired $p$-moment loss ($p=1.2$) minus MSE", "", "| Labels | Seed differences (pp) | Mean delta (pp) | Exact two-sided sign-flip p |", "|---:|---|---:|---:|"]
for n in BUDGETS:
    d = (np.asarray(raw["floc_1.2"][n]) - np.asarray(raw["mse"][n])) * 100
    md.append(f"| {n} | " + ", ".join(f"{x:+.1f}" for x in d) + f" | {d.mean():+.2f} | {exact_signflip_p(d):.4f} |")
md += ["", "This is one fixed configuration. Do not compare its p-values or small differences inferentially against the historical fixed-template table, whose pretraining-seed structure differs."]
(ROOT / "results" / "family_randomized_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
