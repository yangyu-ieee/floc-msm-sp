"""Reconcile archived validation-only Huber/Charbonnier/MCC tuning.

Python stdlib only; no external data, model inference, or GPU.
Run: python revision_audits/audit_validation_selection_ties.py

Validation predictions are reconstructed as integer correct counts from the
five stored per-seed float32 accuracies. No test accuracy is used for ranking.
"""
from __future__ import annotations

import json
from itertools import product
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


def exact_five_seed_signflip(differences: list[int]) -> float:
    """Descriptive two-sided 2^5 sign enumeration on integer test counts."""
    assert len(differences) == 5
    observed = abs(sum(differences))
    return sum(
        abs(sum(s*x for s,x in zip(signs,differences))) >= observed
        for signs in product((-1,1), repeat=5)
    ) / 32


def audit() -> None:
    ties, single_margins = set(), set()
    unique_test_wins, tied_test_wins, test_misses = set(), set(), set()
    p12_contrasts = []
    val_minus_test_counts = []
    retrospective_gap_counts = []
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
            # Test outcomes are used ONLY after reconstructing the validation
            # selection. This retrospective comparison must NEVER select a
            # candidate or update the frozen published benchmark.
            test_counts = {
                v: sum(integer_correct_counts(data[v][n]["test_seeds"]))
                for v in variants
            }
            best_test = max(test_counts.values())
            tied_test = [v for v in variants if test_counts[v] == best_test]
            regret = best_test-test_counts[chosen]
            retrospective_gap_counts.append(regret)
            val_minus_test_counts.append(exact_counts[chosen]-test_counts[chosen])
            key=(n, family)
            if chosen in tied_test:
                if len(tied_test)>1:
                    tied_test_wins.add(key)
                else:
                    unique_test_wins.add(key)
            else:
                test_misses.add(key)

            fixed_p = data["floc_1.2"][n]["test_seeds"]
            chosen_seed = data[chosen][n]["test_seeds"]
            p12_diff = [a-b for a,b in zip(
                integer_correct_counts(fixed_p),
                integer_correct_counts(chosen_seed)
            )]
            p_value=exact_five_seed_signflip(p12_diff)
            assert p_value>=0.0625
            p12_contrasts.append({
                "budget": n, "family": family,
                "delta_pp": sum(p12_diff)/25, "p": p_value,
            })
    assert ties == EXPECTED_TIES, (ties, EXPECTED_TIES)
    assert single_margins == EXPECTED_SINGLE_COUNT_MARGIN, (single_margins, EXPECTED_SINGLE_COUNT_MARGIN)
    print("PASS: 12 archived family/budget selections reproduce recorded val-only ranking.")
    print("Two selections have integer-count ties; three have a one-correct-prediction lead.")
    print("Test accuracies above document evaluation outputs ONLY and are not used for selection.")
    assert len(unique_test_wins)==9, unique_test_wins
    assert tied_test_wins=={("100","charbonnier")}
    assert test_misses=={("500","charbonnier"),("500","mcc")}
    assert sorted(x for x in retrospective_gap_counts if x>0)==[14,15]
    assert len(p12_contrasts)==12
    assert sum(r["delta_pp"]>0 for r in p12_contrasts)==5
    assert sum(r["delta_pp"]<0 for r in p12_contrasts)==6
    assert sum(r["delta_pp"]==0 for r in p12_contrasts)==1
    assert all(r["p"]>=0.125 for r in p12_contrasts)
    assert sum(val_minus_test_counts)==-53  # -53/300 predictions? total 12 x 2500
    print("Retrospective test ranking: 9 unique winners + 1 test tie + 2 misses.")
    print("Both misses are n=500 selections by a 1/2500 validation lead;")
    print("  lost 14/2500 (0.56 pp) and 15/2500 (0.60 pp) test predictions.")
    print(f"Mean test-regret to hindsight-best candidate: "
          f"{sum(retrospective_gap_counts)/len(retrospective_gap_counts)/25:.4f} pp")
    print(f"Chosen validation minus test means, averaged over 12 cells: "
          f"{sum(val_minus_test_counts)/len(val_minus_test_counts)/25:.4f} pp")
    print("Paired p=1.2 minus selected-family test contrasts:")
    for row in p12_contrasts:
        print(f"  n={row['budget']}, {row['family']}: "
              f"{row['delta_pp']:+.2f} pp, signflip p={row['p']:.4f}")
    print("PASS: 5 positive, 6 negative, 1 zero; none has five-pair")
    print("      exact two-sided signflip p<=0.05 (minimum observed 0.125).")
    print("WARNING: testing a validation-selected candidate on the same")
    print("         downstream pipeline seeds remains exploratory; the test")
    print("         outcomes MUST NOT drive retrospective hyperparameter tuning.")


if __name__ == "__main__":
    audit()
