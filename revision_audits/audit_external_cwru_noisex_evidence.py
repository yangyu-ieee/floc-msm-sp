"""26SP R1: NOISEX/UCR and CWRU external-test evidence audit.

Default mode: standard-library-only, read archived v70 JSON and source, and
make NO network requests, GPU runs, or historical-file changes.

Run from a full repo checkout:
    python revision_audits/audit_external_cwru_noisex_evidence.py

NOISEX cell means have 5 end-to-end pipeline repetitions; CWRU has 5 seeds,
but does not reset model/shuffle RNGs per loss. Neither the 16 dataset/shot
cells nor 953 overlapping CWRU windows are independent experimental runs.
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
CW = BASE / "results/cwru"
NX = BASE / "results/noisex"
LOSS_CWRU = ("scratch", "mse", "l1", "huber", "charbonnier", "floc")
LOSS_NOISEX = ("scratch", "mse", "l1", "floc")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(vals):
    return sum(vals) / len(vals)


def exhaustive_signflip(differences):
    """Two-sided paired sum over all 2^n sign allocations, under sign symmetry."""
    observed = abs(sum(differences))
    extreme = sum(
        abs(sum(x*y for x,y in zip(differences, signs))) >= observed - 1e-9
        for signs in product((-1, 1), repeat=len(differences))
    )
    return extreme, 2 ** len(differences), extreme / (2 ** len(differences))


def check_cwru():
    meta = load(CW / "metadata.json")
    data = load(CW / "results.json")
    source = (BASE / "code/cwru_loss_compare_sourceclosed.py").read_text(encoding="utf-8")
    assert meta["source"] == "npz" and meta["channel"] == "DE"
    assert meta["train_loads"] == [0, 1, 2] and meta["test_loads"] == [3]
    assert meta["seq_len"] == 1024 and meta["stride"] == 512
    assert meta["train_shape"] == [2850, 1024]
    assert meta["test_shape"] == [953, 1024]
    assert meta["alpha"] == 1.0 and meta["noise_scale"] == 0.3
    assert meta["shots"] == [10, 20, 50] and meta["seeds"] == 5
    assert meta["losses"] == list(LOSS_CWRU)
    assert len(meta["input_manifest"]) == 16
    file_names = [row["path"].split("/")[-1] for row in meta["input_manifest"]]
    assert len(set(file_names)) == 16
    expected_train = {"1797", "1772", "1750"}
    expected_test = {"1730"}
    trainfiles = {x for x in file_names if x.split("_")[0] in expected_train}
    testfiles = {x for x in file_names if x.split("_")[0] in expected_test}
    assert len(trainfiles) == 12 and len(testfiles) == 4
    assert not (trainfiles & testfiles)
    assert "for seed in range(args.seeds):" in source
    seed_loop = source[source.index("for seed in range(args.seeds):"):]
    assert seed_loop.count("set_seed(seed)") == 1
    assert "weights[lt] = pretrain_msm(x_pt, lt, args, device)" in source
    assert "acc, conf = eval_classifier(" in source
    pretrain = source[source.index("def pretrain_msm("):source.index("def sample_fewshot_indices(")]
    finetune = source[source.index("def eval_classifier("):source.index("def subsample_rows(")]
    assert "set_seed(" not in pretrain and "set_seed(" not in finetune
    assert "shuffle=True" in pretrain and "shuffle=True" in finetune
    assert "rng = np.random.default_rng(seed + 10007 * shots)" in finetune

    counts = None
    for shot in meta["shots"]:
        for loss in LOSS_CWRU:
            key = f"shots={shot}/loss={loss}"
            record = data[key]
            assert len(record["seeds"]) == len(record["confusions"]) == 5
            assert math.isclose(record["mean"], mean(record["seeds"]), abs_tol=1e-12)
            for acc, confusion in zip(record["seeds"], record["confusions"]):
                actual_counts = [sum(row) for row in confusion]
                if counts is None:
                    counts = actual_counts
                assert actual_counts == counts
                correct = sum(confusion[i][i] for i in range(4))
                assert math.isclose(correct / 953, acc, abs_tol=1e-12)
    assert counts == [240, 239, 236, 238]
    majority = max(counts) / sum(counts)
    expected = (("l1", 27.40818468), ("charbonnier", 28.54144806),
                ("floc", 31.10178384))
    for shot, (label, pct) in zip(meta["shots"], expected):
        winner = max(LOSS_CWRU, key=lambda loss: data[f"shots={shot}/loss={loss}"]["mean"])
        observed = 100 * data[f"shots={shot}/loss={winner}"]["mean"]
        assert winner == label and abs(observed-pct) < 1e-5
        print(f"CWRU {shot}-shot: best={winner}, {observed:.3f}%; "
              f"majority={100*majority:.3f}%")
    print("PASS CWRU: 953 test samples, class counts 240/239/236/238;")
    print("  train/test source files disjoint by load, but overlapping windows")
    print("  within each load and loss-wise Torch RNGs are NOT matched.")


def check_noisex():
    meta = load(NX / "run_metadata.json")
    data = load(NX / "results.json")
    summary = load(NX / "summary.json")
    source = (BASE / "code/noisex92_ucr_seeded.py").read_text(encoding="utf-8")
    assert meta["protocol"]["snr_db"] == 5
    assert meta["protocol"]["replication_unit"] == (
        "full pipeline seed, including noise realization and pretraining"
    )
    assert meta["master_seeds"] == [0, 1, 2, 3, 4]
    assert len(meta["datasets"]) == 8
    assert set(meta["noise_file_sha256"]) == {
        "machinegun.wav", "factory1.wav", "factory2.wav"
    }
    assert "x_train_raw, y_train_raw = load_ucr_tsv(dataset, \"TRAIN\")" in source
    assert "x_test_raw, y_test_raw = load_ucr_tsv(dataset, \"TEST\")" in source
    assert "x_pt = make_pretraining_set(" in source
    assert "x_train_noisy, derive_seed(master_seed, dataset, \"pretraining-corpus\")" in source
    assert "seed_all(derive_seed(master_seed, dataset, \"pretrain-model\"))" in source
    assert "all_labels = np.unique(np.concatenate([y_train_raw, y_test_raw]))" in source
    assert "signal_rms * 10 ** (-SNR_DB / 20)" in source
    assert "rng.integers(0, noise_bank.shape[1] - length + 1)" in source
    # Noise bank WAV sample rates are never read; this is a waveform-coordinate
    # stress test, not independently calibrated sensor-noise realism.
    assert "wf.getframerate(" not in source

    datasets = tuple(meta["datasets"])
    cells = [f"{d}/shots={n}" for d in datasets for n in (5, 10)]
    assert len(cells) == summary["n_conditions"] == 16
    assert len(data) == len(cells) * len(LOSS_NOISEX) == 64
    averages = {}
    wins = {name: 0 for name in LOSS_NOISEX}
    for method in LOSS_NOISEX:
        for cell in cells:
            record = data[f"{cell}/loss={method}"]
            assert record["seed_ids"] == list(range(5))
            assert len(record["seeds"]) == 5
            assert math.isclose(record["mean"], mean(record["seeds"]), abs_tol=1e-12)
        averages[method] = 100*mean([data[f"{cell}/loss={method}"]["mean"] for cell in cells])
    for cell in cells:
        leader = max(LOSS_NOISEX, key=lambda m: data[f"{cell}/loss={m}"]["mean"])
        wins[leader] += 1
    assert wins == summary["win_count"] == {
        "scratch": 1, "mse": 5, "l1": 8, "floc": 2
    }
    assert abs(averages["mse"]-58.39172) < 1e-4
    assert abs(averages["l1"]-58.29731) < 1e-4
    assert abs(averages["floc"]-58.17487) < 1e-4
    above = 0
    below = 0
    for ds in datasets:
        delta = mean([
            data[f"{ds}/shots={n}/loss=floc"]["mean"] -
            data[f"{ds}/shots={n}/loss=mse"]["mean"]
            for n in (5,10)
        ])
        above += delta > 0
        below += delta < 0
    assert (above,below) == (4,4)
    print("PASS NOISEX: 8 UCR datasets × 2 shot conditions, five full-pipeline seeds each.")
    print("  Equal-cell averages (%):", {k:round(v, 5) for k,v in averages.items()})
    print("  p=1.2 mean below MSE by %.5f pp; 4/8 datasets positive, 4 negative."
          % (averages["mse"] - averages["floc"]))
    print("  NOISEX is recorded WAV noise sampled in coordinates; no WAV-to-UCR")
    print("  sample-rate or physical-sensor calibration is implemented.")
    print("  Test labels inform class-label vocabulary mapping only; TEST features")
    print("  are not passed into the SSL pretraining dataset.")

    # Data-set is the defensible top-level unit for cross-task summaries.
    # 5-shot and 10-shot outcomes re-use source datasets; five seeds are
    # matched full-pipeline replicates WITHIN each dataset/shot condition.
    # Do NOT treat 16 cells or 80 replicates as 80 independent datasets.
    rows = []
    strict_majority = 0
    zero_strict = []
    distributions = {k: 0 for k in range(6)}
    unique_leaders = 0
    for ds in datasets:
        n_test = int(meta["datasets"][ds]["test_shape"][0])
        record = {"ds": ds, "n_test": n_test}
        for loss in LOSS_NOISEX:
            all_counts = []
            for shot in (5, 10):
                cell = f"{ds}/shots={shot}/loss={loss}"
                vals = data[cell]["seeds"]
                assert data[cell]["seed_ids"] == [0, 1, 2, 3, 4]
                counts = [round(v*n_test) for v in vals]
                assert all(abs(v-c/n_test) < 1e-7 for v,c in zip(vals,counts))
                all_counts.extend(counts)
            record[loss] = 100 * sum(all_counts) / (10*n_test)
        rows.append(record)

        for shot in (5, 10):
            by_loss = {
                loss: [round(v*n_test) for v in
                       data[f"{ds}/shots={shot}/loss={loss}"]["seeds"]]
                for loss in LOSS_NOISEX
            }
            totals = {loss: sum(v) for loss,v in by_loss.items()}
            best_total = max(totals.values())
            mean_winners = [loss for loss in LOSS_NOISEX
                            if totals[loss] == best_total]
            assert len(mean_winners) == 1, (ds, shot, mean_winners)
            unique_leaders += 1
            leader = mean_winners[0]
            strict = sum(
                all(by_loss[leader][i] > by_loss[other][i]
                    for other in LOSS_NOISEX if other != leader)
                for i in range(5)
            )
            distributions[strict] += 1
            strict_majority += strict >= 3
            if strict == 0:
                zero_strict.append((ds, shot))
            if ds == "ECG200" and shot == 5:
                assert leader == "l1" and strict == 0
                assert totals["l1"] == 344 and totals["floc"] == 342
            if ds == "FaceAll" and shot == 10:
                assert leader == "l1" and strict == 5

    assert unique_leaders == 16
    assert strict_majority == 7
    assert zero_strict == [("ECG200",5)]
    assert distributions == {0: 1, 1: 5, 2: 3, 3: 5, 4: 1, 5: 1}
    print("NOISEX mean-winner stability across five full-pipeline seeds:")
    print("  Only 7/16 condition-average winners strictly win >=3/5 seeds.")
    print("  ECG200/5-shot mean winner L1 wins ZERO of its individual seeds;")
    print("  FaceAll/10-shot L1 strictly wins all five full-pipeline seeds.")

    # Each dataset effect is first averaged across its two shot budgets;
    # 8 datasets, exhaustive 2^8 sign changes. P-values are exploratory:
    # source tasks are selected, not randomly sampled; formal exactness
    # would require paired-effect sign exchangeability under the null.
    expected = {
        "mse": (-0.2168492530460444, 4, 4, 192),
        "l1": (-0.12243129705377243, 3, 5, 202),
        "scratch": (+6.090535257026891, 7, 1, 4),
    }
    for baseline, (exp_mean, exp_wins, exp_losses, exp_ext) in expected.items():
        effects = [z["floc"]-z[baseline] for z in rows]
        extremal, total, pvalue = exhaustive_signflip(effects)
        assert total == 256 and extremal == exp_ext
        assert math.isclose(mean(effects), exp_mean, abs_tol=1e-9)
        assert (sum(e > 1e-9 for e in effects),
                sum(e < -1e-9 for e in effects)) == (exp_wins, exp_losses)
        print(f"  p=1.2 minus {baseline:>7}: {mean(effects):+.4f} pp,"
              f" datasets +/- {exp_wins}/{exp_losses}, "
              f"signflip {extremal}/256={pvalue:.6f}")
    print("  Exploratory Holm over two robust comparator tests (MSE/L1):")
    print("  both adjusted p=1.0. Scratch comparison is not included: ")
    print("  scratch gets 80 supervised epochs, pretrained methods 50.")
    print("  Sign-flip enumeration is NOT population-level statistical proof.")


if __name__ == "__main__":
    check_cwru()
    check_noisex()
