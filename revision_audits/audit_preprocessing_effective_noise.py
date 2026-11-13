"""Reconcile 26SP R1 Table 2 and deterministic effective-noise scaling.

Standard-library archived-record check. Does not retrain models or use test data
for model selection. Run from repo root:
    python revision_audits/audit_preprocessing_effective_noise.py

No formal hypothesis tests are claimed from the 8 correlated cells x 5 seeds.
"""
from __future__ import annotations

import json
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
DATA = ROOT / "results/normalization"
LOSSES = ("mse", "l1", "charbonnier", "floc_1.2")
BUDGETS = ("100", "200")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    src = (ROOT / "code/r5_preprocessing_audit.py").read_text(encoding="utf-8")
    cfg = load(DATA / "config.json")
    summary = load(DATA / "summary.json")
    diag = load(DATA / "input_diagnostics.json")
    assert cfg["script_sha256"] == "5e2655170dbbeb54771fc4c473784cd27e6765dd5641ef51cc6c4fcfc6434cf9"
    assert cfg["alpha"] == 1.5 and cfg["seeds"] == 5
    assert cfg["shots"] == [100, 200] and cfg["test_size"] == 500
    assert "clean.astype(np.float64) / train_scale + noise_scale * noise" in src
    assert "observed /= np.max(np.abs(observed), axis=1, keepdims=True) + 1e-8" in src
    assert "pt_vars = noise_variates(clean_pt.shape, seed=13002)" in src
    assert "train_vars = noise_variates(clean_train.shape, seed=40000 + seed)" in src
    assert "test_vars = noise_variates(clean_test.shape, seed=50000 + seed)" in src

    s_pt = diag["pretraining"]["clean_train_global_rms"]
    assert isclose(s_pt, 0.625311415410149, abs_tol=1e-12)
    scale_values = []
    for budget in BUDGETS:
        for seed in range(5):
            val = diag["downstream"][budget][str(seed)]["cms_clean_global_rms"]["clean_train_global_rms"]
            assert val > 0
            scale_values.append(val)
    assert isclose(min(scale_values), 0.7181492518477215, abs_tol=1e-12)
    assert isclose(max(scale_values), 0.7191069393438372, abs_tol=1e-12)

    positives = 0
    zeros = 0
    negatives = 0
    deltas = []
    for loss in LOSSES:
        for budget in BUDGETS:
            arm_max = summary["cms_maxabs"][loss][budget]
            arm_rms = summary["cms_clean_global_rms"][loss][budget]
            a, b = arm_max["seeds"], arm_rms["seeds"]
            assert len(a) == len(b) == 5
            differences = [100*(y-x) for x,y in zip(a,b)]
            assert isclose(sum(differences)/5, 100*(arm_rms["mean"]-arm_max["mean"]), abs_tol=1e-8)
            deltas.append(sum(differences)/5)
            positives += sum(d > 1e-7 for d in differences)
            zeros += sum(abs(d) <= 1e-7 for d in differences)
            negatives += sum(d < -1e-7 for d in differences)
            print(f"{loss:>11}, labels={budget}: mean delta={sum(differences)/5:+.2f} pp; "
                  f"positive/zero/negative seeds={sum(d>1e-7 for d in differences)}/"
                  f"{sum(abs(d)<=1e-7 for d in differences)}/"
                  f"{sum(d < -1e-7 for d in differences)}")
    assert (positives,zeros,negatives) == (39,1,0)
    assert isclose(min(deltas),1.88,abs_tol=1e-4) and isclose(max(deltas),3.64,abs_tol=1e-4)
    assert isclose(sum(deltas)/len(deltas),2.86,abs_tol=1e-4)

    maxabs = diag["pretraining"]["cms_maxabs"]
    rms = diag["pretraining"]["cms_clean_global_rms"]
    assert isclose(maxabs["abs_max"], 1,abs_tol=1e-12)
    assert isclose(rms["abs_max"],5028.859375,abs_tol=1e-9)

    print(f"Pretraining clean RMS S = {s_pt:.6f}; relative raw-noise scale factor S = {100*s_pt:.2f}%")
    print(f"Downstream S range = {min(scale_values):.6f}...{max(scale_values):.6f};"
          " relative-noise scale ratio 71.81%-71.91%")
    print("Algebra: (x/S + sigma*z)*S = x + (S*sigma)*z.")
    print("These are amplitude-scale ratios, not variance or physical SNR for alpha<2 stable noise.")
    print(f"Mean paired accuracy delta across 8 protocol cells = {sum(deltas)/len(deltas):.2f} pp")
    print(f"Positive/equal/negative paired differences = {positives}/{zeros}/{negatives} "
          "(correlated cells and shared pretraining checkpoints, not 40 independent trials)")
    print("PASS: archived protocol, input diagnostics and Table 2 means are consistent.")


if __name__ == "__main__":
    main()
