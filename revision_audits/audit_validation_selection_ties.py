"""Reconcile archived validation-only Huber/Charbonnier/MCC tuning.

Python stdlib only; no external data, model inference, or GPU.
Run: python revision_audits/audit_validation_selection_ties.py

Validation predictions are reconstructed as integer correct counts from the
five stored per-seed float32 accuracies. No test accuracy is used for ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "revision_v70_20261010/results"
FAMILIES = {
    "huber": ("huber_0.25", "huber_0.5", "huber_1.0", "huber_2.0"),
    "charbonnier": ("charbonnier_0.001", "charbonnier_0.01", "charbonnier_0.1"),
    "mcc": ("mcc_0.1", "mcc_0.25", "mcc_0.5"),
}
EXAMPLES_PER_SEED = 500
EXPECTED_TIES = {("100", "charbonnier"), ("200", "mcc")}
EXPECTED_SINGLE_COUNT_MARGIN = {("200", "huber"), ("500", "charbonnier"), ("500", "mcc")}


def load(folder: str, filename: str):
    return json.loads((ROOT / folder / filename).read_text(encoding="utf-8"))


def integer_correct_counts(seed_accuracy: list[float]) -> list[int]:
    assert len(seed_accuracy) == 5, len(seed_accuracy)
    counts = [round(x * EXAMPLES_PER_SEED) for x in seed_accuracy]
    assert all(0 <= c <= EXAMPLES_PER_SEED for c in counts)
    assert all(abs(x - c / EXAMPLES_PER_SEED) < 1e-7
               for x, c in zip(seed_accuracy, counts))
    return counts


def audit() -> None:
    ties, single_margins = set(), set()
    print("N | family | selected | tied-best-candidate count | correct-count gap | test accuracy (selected)")
    for n in ("50", "100", "200", "500"):
        folder = "validation_selected_n100" if n == "100" else "validation_selected"
        data, selection = load(folder, "results.json"), load(folder, "validation_selection.json")
        cfg = load(folder, "config.json")
        assert int(n) in cfg["n_labeled"]
        assert cfg["n_test"] == EXAMPLES_PER_SEED and cfg["n_seeds"] == 5
        for family, variants in FAMILIES.items():
            raw_scores = {variant: data[variant][n]["val_mean"] for variant in variants}
            exact_counts = {variant: sum(integer_correct_counts(data[variant][n]["val_seeds"]))
                            for variant in variants}
            # Runner selection is raw-float max. This reproduces its implementation,
            # but does not imply an actual difference when integer correct counts tie.
            chosen = max(variants, key=raw_scores.get)
            assert selection[n][family]["selected_variant"] == chosen
            assert abs(selection[n][family]["heldout_test_mean"] - data[chosen][n]["test_mean"]) < 1e-12
            count_order = sorted(exact_counts.values(), reverse=True)
            assert exact_counts[chosen] == count_order[0], (n, family, chosen)
            gap = count_order[0] - count_order[1]
            ntied = sum(v == count_order[0] for v in exact_counts.values())
            if gap == 0:
                ties.add((n, family))
            elif gap == 1:
                single_margins.add((n, family))
            print(f"{n:>3} | {family:>11} | {chosen:>18} | {ntied} | {gap:>2} | "
                  f"{data[chosen][n]['test_mean']*100:.2f}%")
    assert ties == EXPECTED_TIES, (ties, EXPECTED_TIES)
    assert single_margins == EXPECTED_SINGLE_COUNT_MARGIN, (single_margins, EXPECTED_SINGLE_COUNT_MARGIN)
    print("PASS: 12 archived family/budget selections reproduce recorded val-only ranking.")
    print("Two selections have integer-count ties; three have a one-correct-prediction lead.")
    print("Test accuracies above document evaluation outputs ONLY and are not used for selection.")


if __name__ == "__main__":
    audit()
