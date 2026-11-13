"""Archived 26SP R1 Student-t sensitivity: source protocol and exact five-pair signs.

Python 3.10+ standard library only. Run at repository root:
    python revision_audits/audit_student_t_protocol_and_signflip.py

No machine-learning imports, downloads, GPU computation or retraining.
Sign-flip p values enumerate all 2**5 configurations; formal exact null
inference additionally requires the appropriate sign-symmetry/exchangeability.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
STUDENT_SRC = BASE / "code/floc_msm_student_t.py"
STABLE_SRC = BASE / "code/floc_msm_main_corrected.py"
CONFIG = BASE / "results/student_t/config.json"
RESULTS = BASE / "results/student_t/results.json"
BUDGETS = ("50", "200", "500")
METHODS = ("scratch", "mse", "l1", "huber", "charbonnier", "floc_1.2")
EXPECTED_GAIN_PP = (1.92, 3.08, 1.84)
EXPECTED_BEST = ("charbonnier", "charbonnier", "l1")
EXPECTED_LEADER_MARGIN_COUNTS = (6, 6, 4)


def enumerate_signflip(differences: list[float]) -> float:
    assert len(differences) == 5
    observed = abs(sum(differences))
    extreme = sum(
        abs(sum(sign * diff for sign, diff in zip(signs, differences)))
        >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=5)
    )
    return extreme / 32


def count_correct(accuracy: float, test_n: int) -> int:
    n = round(test_n * accuracy)
    assert abs(n / test_n - accuracy) < 1e-7, (accuracy, test_n)
    return n


def check_source() -> None:
    student = STUDENT_SRC.read_text(encoding="utf-8")
    stable = STABLE_SRC.read_text(encoding="utf-8")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["student_t_nu"] == 1.5
    assert cfg["pretraining_noise_scale"] == 0.15
    assert cfg["downstream_noise_scale"] == 0.30
    assert cfg["n_seeds"] == 5 and cfg["n_test"] == 500
    assert cfg["n_labeled"] == [50, 200, 500]
    assert cfg["losses"] == list(METHODS)
    assert "np.random.standard_t(args.nu,size=shape)*scale" in student
    assert "student_t_noise(seq_len,0.15)" in student
    assert "student_t_noise(seq_len,0.30)" in student
    assert "X_pt=gen_unlabeled(CONFIG['pt_samples'],128,True)" in student
    assert "Xtr,ytr=gen_cls(n_train,128,True); Xte,yte=gen_cls(n_test,128,True)" in student
    assert "torch.manual_seed(args.seed + 1)" in student
    assert "generator=torch.Generator().manual_seed(args.seed + 2)" in student
    assert student.count("torch.manual_seed(seed); np.random.seed(seed)") >= 2
    assert "CONFIG['ft_epochs'] if n_labeled<=200 else max(CONFIG['ft_epochs'],50)" in student
    # Student-t masking: per-sequence Binomial(32,0.5), not exact 16-of-32.
    assert "mask=torch.rand(xb.shape[0],tok.shape[1],1,device=device)<0.5" in student
    assert "random_order=torch.rand(xb.shape[0],tok.shape[1],1,device=device).argsort(dim=1)" in stable
    # Separate streams: the distribution generators consume different NumPy
    # random draws interleaved with clean waveform/jitter synthesis.
    assert "st=np.random.randint(0,5)" in student and "st=np.random.randint(0,5)" in stable
    assert "np.random.standard_t" in student and "draw_cms_variates" in stable
    print("PASS: Student-t nu=1.5 and scales 0.15/0.30 used for pretraining and downstream.")
    print("PASS: same-seed loss controls within Student-t; one checkpoint per loss.")
    print("CAUTION: Bernoulli(0.5) mask count is Binomial(32,0.5), SD=sqrt(8),")
    print("         whereas corrected alpha=1.5 core masks exactly 16 of 32 tokens.")
    print("CAUTION: Student-t vs stable core does not hold clean/noise samples fixed.")


def audit_results() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert set(results) == set(METHODS)
    for budget, expected_gain, best, expected_lead in zip(BUDGETS, EXPECTED_GAIN_PP, EXPECTED_BEST, EXPECTED_LEADER_MARGIN_COUNTS):
        for method in METHODS:
            vals = results[method][budget]["seeds"]
            assert len(vals) == 5
            acc_mean = sum(vals) / 5
            std = math.sqrt(sum((x - acc_mean)**2 for x in vals) / 5)
            assert abs(acc_mean - results[method][budget]["mean"]) < 1e-10
            assert abs(std - results[method][budget]["std"]) < 1e-10
            assert all(count_correct(x, cfg["n_test"]) >= 0 for x in vals)
        winner = max(METHODS, key=lambda m: results[m][budget]["mean"])
        assert winner == best, (budget, winner, best)
        pooled_correct = {
            method: sum(count_correct(val, cfg["n_test"])
                        for val in results[method][budget]["seeds"])
            for method in METHODS
        }
        descending = sorted(pooled_correct.values(), reverse=True)
        assert descending[0] - descending[1] == expected_lead, (
            budget, pooled_correct, expected_lead
        )
        p12 = results["floc_1.2"][budget]["seeds"]
        mse = results["mse"][budget]["seeds"]
        differences = [x-y for x, y in zip(p12, mse)]
        gain_pp = 100 * sum(differences) / 5
        assert abs(gain_pp - expected_gain) < 1e-4
        assert all(d > 1e-8 for d in differences)
        p = enumerate_signflip(differences)
        assert abs(p-0.0625) < 1e-12
        gains_count = [count_correct(x, cfg["n_test"]) -
                       count_correct(y, cfg["n_test"]) for x, y in zip(p12, mse)]
        print(f"Student-t nu=1.5, n={budget}: top={winner}, "
              f"p=1.2 vs MSE={gain_pp:+.2f} pp, wins=5/5, "
              f"exhaustive sign-flip p={p:.4f}, correct-count diffs={gains_count}")
    print("PASS: 18 loss/budget cells; 5-seed means, SDs and rankings.")
    print("PASS: top-two margins are 6/2500, 6/2500, 4/2500 correct.")
    print("CAUTION: 500 labels also gets 50 FT epochs vs 30 for 50/200.")
    print("Do NOT pool the 15 seed-by-budget differences as independent experiments.")


if __name__ == "__main__":
    check_source()
    audit_results()
