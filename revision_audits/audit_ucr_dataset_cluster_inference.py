"""Archived UCR: dataset-cluster exploratory sign flips and metric weighting.

Run from repository root:
    python revision_audits/audit_ucr_dataset_cluster_inference.py

Standard library only. No model training, downloads, random permutations,
or changes to the frozen archive. UCR's 5-shot and 10-shot conditions are
clustered by DATASET; their fixed test records are repeatedly evaluated.
Sign flips have formal inferential interpretation only under sign symmetry
of the across-dataset paired effects. These ten datasets are not a random
sample of a defined population.
"""
from __future__ import annotations

import itertools
import json
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
D = ROOT / "results/ucr_corrected"
BASELINES = ("mse", "l1", "huber", "charbonnier")
EXPECTED_EXACT_TAIL_COUNTS = {
    "mse": 60,
    "l1": 1012,
    "huber": 36,
    "charbonnier": 696,
}


def load(name: str):
    return json.loads((D / name).read_text(encoding="utf-8"))


def mean(xs):
    return sum(xs) / len(xs)


def signflip(x):
    """Two-sided absolute-mean statistic, all 2^k sign assignments."""
    observed = abs(sum(x))
    extreme = sum(
        abs(sum(v*s for v,s in zip(x, signs))) >= observed - 1e-9
        for signs in itertools.product((-1, 1), repeat=len(x))
    )
    return extreme, 2 ** len(x), extreme / 2 ** len(x)


def holm(values):
    indexed = sorted(enumerate(values), key=lambda r: r[1])
    adj = [None] * len(values)
    prev = 0.0
    for rank, (i, p) in enumerate(indexed):
        prev = max(prev, (len(values) - rank) * p)
        adj[i] = min(1.0, prev)
    return adj


def audit():
    cfg, data, diagnostics, archive_summary = (
        load("config.json"), load("results.json"),
        load("dataset_diagnostics.json"), load("summary.json"))
    assert len(cfg["datasets"]) == 10
    assert cfg["shots"] == [5,10] and cfg["n_seeds"] == 5
    assert cfg["losses"] == ["scratch", "mse", "l1",
                             "huber", "charbonnier", "floc"]
    assert archive_summary["total"] == 20
    assert archive_summary["win_count"] == {
        "scratch": 2, "mse": 5, "l1": 5,
        "huber": 1, "charbonnier": 4, "floc": 3}

    runner = (ROOT / "code/ucr_all_losses_corrected.py").read_text(
        encoding="utf-8")
    assert "X_train_np,y_train_np=load_classification(ds_name,split='train')" in runner
    assert "X_test_np,y_test_np=load_classification(ds_name,split='test')" in runner
    assert "X_pt=np.tile(X_train_n,(n_aug,1))" in runner
    assert "for shots in SHOTS:" in runner
    assert "for seed in range(N_SEEDS):" in runner
    assert "np.random.seed(seed); torch.manual_seed(seed)" in runner

    rows = []
    for ds in cfg["datasets"]:
        n_test = diagnostics[ds]["test_shape"][0]
        assert n_test >= 1
        row = {"dataset": ds, "n_test": n_test}
        for loss in cfg["losses"]:
            correct_total = 0
            for shot in cfg["shots"]:
                key = f"{ds}/shots={shot}/loss={loss}"
                record = data[key]
                assert len(record["seeds"]) == cfg["n_seeds"]
                assert isclose(
                    record["mean"], mean(record["seeds"]),
                    abs_tol=5e-7)
                for accuracy in record["seeds"]:
                    count = round(accuracy * n_test)
                    assert 0 <= count <= n_test
                    assert abs(accuracy-count/n_test) < 5e-7, (key,accuracy)
                    correct_total += count
            # Percentage mean over two shot conditions and five downstream
            # seeds, ALL on the same n_test fixed official test examples.
            row[loss] = correct_total * 100.0 / (
                len(cfg["shots"]) * cfg["n_seeds"] * n_test)
        rows.append(row)

    macro = {
        loss: mean([row[loss] for row in rows])
        for loss in cfg["losses"]
    }
    n_sum = sum(row["n_test"] for row in rows)
    assert n_sum == 20318
    test_weighted = {
        loss: sum(row[loss] * row["n_test"] for row in rows) / n_sum
        for loss in cfg["losses"]
    }
    assert isclose(macro["mse"], 55.911849763718386, abs_tol=1e-8)
    assert isclose(macro["floc"], 57.02202427857016, abs_tol=1e-8)
    assert isclose(macro["l1"], 57.012220555827966, abs_tol=1e-8)
    assert isclose(macro["charbonnier"], 57.19143515935036, abs_tol=1e-8)
    assert isclose(test_weighted["l1"], 61.43665715129442, abs_tol=1e-8)
    assert isclose(test_weighted["floc"], 61.005512353578105, abs_tol=1e-8)
    assert max(macro, key=macro.get) == "charbonnier"
    assert max(test_weighted, key=test_weighted.get) == "l1"
    print("Dataset-equal macro accuracy and alternative test-size weighting:")
    for loss in cfg["losses"]:
        print(f"  {loss:>12}: macro={macro[loss]:.4f}%, "
              f"test-size weighted={test_weighted[loss]:.4f}%")
    print("NOTE: alternate weighting changes the descriptive estimand;")
    print("      it DOES NOT turn reused 5-seed/2-shot test records into")
    print("      independent test observations.")

    names, pvals = [], []
    print("Floc p=1.2 minus baseline: clustered by each dataset's 5/10-shot mean")
    for baseline in BASELINES:
        diff = [row["floc"] - row[baseline] for row in rows]
        ext, total, p = signflip(diff)
        assert total == 1024 and ext == EXPECTED_EXACT_TAIL_COUNTS[baseline]
        names.append(baseline)
        pvals.append(p)
        wins = sum(d > 1e-7 for d in diff)
        losses = sum(d < -1e-7 for d in diff)
        assert wins+losses==10
        deleted_means = [mean(diff[:i] + diff[i+1:]) for i in range(10)]
        if baseline == "mse":
            assert wins == 7 and losses == 3
            assert isclose(mean(diff), 1.1101745148517748, abs_tol=1e-9)
            assert isclose(min(deleted_means), 0.8285889671192557, abs_tol=1e-9)
            assert isclose(max(deleted_means), 1.3890827942797488, abs_tol=1e-9)
        if baseline == "l1":
            assert wins == 5 and losses == 5
        print(f"  {baseline:>12}: mean_diff={mean(diff):+.4f}pp, "
              f"datasets +/−={wins}/{losses}, "
              f"two-sided enumeration={ext}/{total} = {p:.8f}; "
              f"leave-one-dataset-out range="
              f"[{min(deleted_means):+.4f},{max(deleted_means):+.4f}]pp")
    adjusted = holm(pvals)
    expected = {"mse": 0.17578125, "l1": 1.0,
                "huber": 0.140625, "charbonnier": 1.0}
    assert all(isclose(adjusted[i],expected[name],abs_tol=1e-12)
               for i,name in enumerate(names))
    print("Holm-adjusted exploratory p across four fixed-baseline contrasts:")
    for name,p in zip(names,adjusted):
        print(f"  {name:>12}: {p:.8f}")
    assert all(p >= .05 for p in adjusted)
    print("PASS: no four-family-Holm adjusted p<0.05.")
    print("Do not claim dataset-sampled population inference, statistical")
    print("equivalence, or independent repetition across 20 shot conditions.")


if __name__ == "__main__":
    audit()
