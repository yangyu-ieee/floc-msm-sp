"""26SP R1: randomized-family task scope and integer-accurate sign-flip audit.

Run from the full repository checkout (Python standard library only):
  python revision_audits/audit_family_randomized_task_scope.py

Does not retrain, fetch datasets, edit original archived results, or modify
submitted manuscripts.  Each archived seed_*/results.json holds one evaluation
seed per  independently pretrained loss checkpoint.
"""
from __future__ import annotations

from itertools import product
import hashlib
import json
from math import isclose
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
SOURCE = ROOT / "code/floc_msm_randomized_class_audit.py"
DATA = ROOT / "results/family_randomized"
SUMMARY = ROOT / "results/family_randomized_summary.json"
LOSSES = ("scratch", "mse", "l1", "huber", "charbonnier", "floc_1.2")
BUDGETS = (50, 100, 200, 500)
TEST_SIZE = 500


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ncorrect(value):
    count = round(value * TEST_SIZE)
    assert abs(value - count / TEST_SIZE) < 1e-7, value
    return count


def exact_signflip_int(diffs):
    """Exact finite-enumeration statistic in integer correct-prediction units."""
    assert len(diffs) == 5
    observed = abs(sum(diffs))
    count = sum(
        abs(sum(s * d for s, d in zip(signs, diffs))) >= observed
        for signs in product((-1, 1), repeat=len(diffs))
    )
    return count / 32


def audit():
    source = SOURCE.read_text(encoding="utf-8")
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    configs = [read(DATA / f"seed_{seed}" / "config.json")
               for seed in range(5)]
    archive = [read(DATA / f"seed_{seed}" / "results.json")
               for seed in range(5)]
    summary = read(SUMMARY)
    assert all(c["code_sha256"] == sha for c in configs)
    assert [c["seed"] for c in configs] == list(range(5))
    assert all(c["n_seeds"] == 1 and c["class_mode"] == "family_randomized"
               and c["mask_strategy"] == "random_token"
               and c["mask_count_rule"].startswith("exactly round")
               for c in configs)
    assert all(c["n_test"] == TEST_SIZE and c["n_labeled"] == list(BUDGETS)
               and c["losses"] == list(LOSSES) for c in configs)
    assert all(c["ft_epochs"] == 30 and c["scratch_epochs"] == 80
               for c in configs)
    assert summary["task"] == "family_randomized"
    assert summary["seeds"] == 5
    assert summary["pretraining_per_loss_seed"] == 1
    assert summary["fine_tune_seeds_per_checkpoint"] == 1

    assert "st=np.random.randint(0,5)" in source
    assert "if CONFIG['class_mode']=='family_randomized':" in source
    assert "cls=i%5" in source
    assert "X_pt=gen_unlabeled(CONFIG['pt_samples'],128,CONFIG['alpha'])" in source
    assert "Xtr,ytr=gen_cls(n_train,128,1.5); Xte,yte=gen_cls(n_test,128,1.5)" in source
    assert "torch.manual_seed(args.seed + 1)" in source
    assert "generator=torch.Generator().manual_seed(args.seed + 2)" in source
    assert "run_seed=args.seed+seed" in source
    assert "torch.manual_seed(run_seed); np.random.seed(run_seed)" in source
    assert "CONFIG['ft_epochs'] if n_labeled<=200 else max(CONFIG['ft_epochs'],50)" in source
    print("PASS: five separately seeded SSL + downstream pipeline runs.")
    print("Family identities are seen in SSL pretraining; test uses new parameter")
    print("draws from the SAME within-family distribution as labeled training.")
    print("This is NOT novel-family OOD recognition or a fixed-model transfer test.")

    observed_mismatches = []
    expected_leads = {50: 5, 100: 6, 200: 24, 500: 1}
    expected_winner = {50: "charbonnier", 100: "mse",
                       200: "l1", 500: "charbonnier"}

    for budget in BUDGETS:
        counts_by_loss = {}
        for loss in LOSSES:
            scores = [r[loss][str(budget)]["seeds"][0] for r in archive]
            assert all(len(r[loss][str(budget)]["seeds"]) == 1 for r in archive)
            counts = [ncorrect(v) for v in scores]
            counts_by_loss[loss] = counts
            values = [v / TEST_SIZE for v in counts]
            stored = summary["losses"][loss][str(budget)]
            assert abs(statistics.mean(values) - stored["mean"]) < 1e-7
            assert abs(statistics.pstdev(values) - stored["sd_population"]) < 1e-7
            assert all(abs(a-b) < 1e-7
                       for a, b in zip(values, stored["seed_accuracies"]))

        ranking = sorted(
            LOSSES, key=lambda loss: sum(counts_by_loss[loss]), reverse=True
        )
        lead = sum(counts_by_loss[ranking[0]]) - sum(counts_by_loss[ranking[1]])
        assert ranking[0] == expected_winner[budget]
        assert lead == expected_leads[budget]
        print(f"n={budget}, top={ranking[0]}, runnerup={ranking[1]}, "
              f"margin={lead} additional correct out of 2500")

        for loss in LOSSES:
            if loss == "mse":
                continue
            differences = [x-y for x,y in zip(
                counts_by_loss[loss], counts_by_loss["mse"]
            )]
            p_int = exact_signflip_int(differences)
            p_old = summary["losses"][loss][str(budget)][
                "paired_exact_signflip_p_vs_mse"
            ]
            if not isclose(p_int, p_old, abs_tol=1e-12):
                observed_mismatches.append((loss, budget, p_old, p_int))
            expected_delta = sum(differences)/25
            assert abs(expected_delta - summary["losses"][loss][str(budget)][
                "delta_vs_mse_pp"]) < 1e-4

        if budget == 100:
            scratch = counts_by_loss["scratch"]
            mse = counts_by_loss["mse"]
            assert [scratch[0], scratch[2]] == [100,100]
            gains = [v-u for v,u in zip(mse,scratch)]
            assert gains == [117,41,134,78,97]
            assert sum(gains) == 467
            print("100 labels: scratch seed0/2 = 20.0% exactly;")
            print("  MSE minus scratch = 18.68 pp overall.")
            print("  Two chance-level scratch seeds account for 10.04 pp")
            print("  (53.75% of the total mean gain), but no confusion")
            print("  matrices exist to diagnose prediction collapse.")

    expected_mismatch = ("charbonnier", 50, 0.8125, 0.875)
    assert len(observed_mismatches) == 1
    mismatch = observed_mismatches[0]
    assert mismatch[0:2] == expected_mismatch[0:2]
    assert isclose(mismatch[2], expected_mismatch[2], abs_tol=1e-12)
    assert isclose(mismatch[3], expected_mismatch[3], abs_tol=1e-12)
    print("ARCHIVED FLOAT-ROUNDING DISCREPANCY:", mismatch)
    print("Float32 accuracy enumeration: p=0.8125; integer-count exact p=0.8750.")
    print("Correction is audit-only. No statistical significance conclusion changes.")
    print("PASS: source/config/statistics checked; no historical files changed.")


if __name__ == "__main__":
    audit()
