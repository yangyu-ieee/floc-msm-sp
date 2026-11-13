"""Audit the single archived learnable-p run without retraining.

Run from repository root:
    python revision_audits/audit_learnable_p_scale_and_provenance.py

Uses only Python standard library, frozen v70 code/JSON and arithmetic.
The p-trajectory *.npy is absent from the frozen archive; this audit
deliberately DOES NOT infer a final learned p from its loss trajectory.
"""
from __future__ import annotations

from itertools import product
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
CODE = ROOT / "code/floc_msm_main_corrected.py"
CONFIG = ROOT / "results/main_alpha15/config.json"
RESULTS = ROOT / "results/main_alpha15/results.json"
TRAINING = ROOT / "results/main_alpha15/pretraining_trajectory_floc_learnable.json"
MANIFEST = ROOT / "MANIFEST_SHA256.txt"
P_TRACE = ROOT / "results/main_alpha15/p_trajectory_floc_learnable.npy"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def exact_signflip_integer(differences):
    assert len(differences) == 5
    observation = abs(sum(differences))
    return sum(
        abs(sum(sign * delta for sign, delta in zip(signs, differences)))
        >= observation
        for signs in product((-1, 1), repeat=5)
    ) / 32


def audit() -> None:
    source = CODE.read_text(encoding="utf-8")
    cfg = read_json(CONFIG)
    result = read_json(RESULTS)
    trajectory = read_json(TRAINING)
    manifest = MANIFEST.read_text(encoding="utf-8")
    observed_hash = hashlib.sha256(CODE.read_bytes()).hexdigest()
    assert observed_hash == cfg["code_sha256"]
    assert observed_hash == (
        "880fbd11139d81d5abe462cd5190a28c973dee1c54e7932d7f629a9c7b541104"
    )
    assert observed_hash in manifest
    assert cfg["alpha"] == 1.5 and cfg["pt_epochs"] == 20
    assert cfg["n_seeds"] == 5 and cfg["n_test"] == 500
    assert cfg["losses"].count("floc_learnable") == 1

    assert "p_raw=nn.Parameter(torch.tensor(" in source
    assert "params.append(p_raw)" in source
    assert "p=0.5+1.5*torch.sigmoid(p_raw)" in source
    assert "loss=r.abs().pow(p).mean()" in source
    assert "opt=torch.optim.Adam(params,lr=1e-3)" in source
    assert "torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0)" in source
    assert "result['_p_history']=p_history" in source
    assert "np.save(f'{LOG_DIR}/p_trajectory_{loss_type}.npy'" in source
    assert "torch.manual_seed(args.seed + 1)" in source
    assert "generator=torch.Generator().manual_seed(args.seed + 2)" in source

    raw_initial = math.log((1.5-0.5)/(2.0-1.5))
    p_initial = 0.5 + 1.5/(1+math.exp(-raw_initial))
    assert math.isclose(p_initial, 1.5, abs_tol=1e-12)
    assert 0.5 < p_initial < 2.0
    assert (0.5 < 1.0 < 1.5 < 2.0)
    # The loss-specific p derivative is unregularized:
    # d |r|^p / dp = |r|^p * ln(|r|), which changes sign under unit rescaling.
    def objective_p_derivative(r, exponent):
        x = abs(r)
        return 0.0 if x == 0 else x**exponent * math.log(x)
    assert objective_p_derivative(0.5, p_initial) < 0
    assert objective_p_derivative(2.0, p_initial) > 0
    assert objective_p_derivative(4.0*0.5, p_initial) > 0
    print("PASS: p participates in Adam and starts at 1.5, bounded to (0.5,2).")
    print("PASS: raw loss gradients favor larger p for abs(r)<1 and smaller p")
    print("      for abs(r)>1; changing signal units can reverse the direction.")
    print("NOTE: no p-specific constraint p<alpha, validation objective, or p-penalty.")

    assert len(trajectory) == 20
    first = trajectory[0]["mean_train_loss"]
    last = trajectory[-1]["mean_train_loss"]
    assert math.isclose(first, 0.1034750947846642, abs_tol=1e-12)
    assert math.isclose(last, 0.04840549784181993, abs_tol=1e-12)
    print("Learnable-p train objective: %.6f -> %.6f (%+.2f%%)" %
          (first, last, 100*(last/first-1)))
    print("Do NOT interpret objective decrease alone as representation learning,")
    print("because p and the loss function itself both change.")

    assert not P_TRACE.exists()
    assert not any("p_trajectory_" in line for line in manifest.splitlines())
    print("ARCHIVAL GAP: p_trajectory_floc_learnable.npy not present in frozen v70.")
    print("Therefore NO source-verified final or epochwise learned-p values exist.")

    for budget in ("50", "100", "200", "500"):
        learned = result["floc_learnable"][budget]
        fixed = result["floc_1.2"][budget]
        a = [round(x*500) for x in learned["seeds"]]
        b = [round(x*500) for x in fixed["seeds"]]
        assert len(a) == len(b) == 5
        diffs = [x-y for x,y in zip(a,b)]
        for x, raw in zip(a, learned["seeds"]):
            assert math.isclose(x/500, raw, abs_tol=1e-7)
        for x, raw in zip(b, fixed["seeds"]):
            assert math.isclose(x/500, raw, abs_tol=1e-7)
        delta_pp = sum(diffs)/25  # 5 seeds x 500 tests: 100/2500=0.04 pp
        assert math.isclose(delta_pp, 100*(learned["mean"]-fixed["mean"]),
                            abs_tol=1e-4)
        assert delta_pp < 0
        p_value = exact_signflip_integer(diffs)
        print(f"n={budget}: learnable-p minus fixed p=1.2 "
              f"= {delta_pp:+.2f} pp, signflip p={p_value:.4f}, "
              f"count diffs={diffs}")
    # Pure-math checks under the *hypothetical non-degenerate* S_alpha S
    # residual R, using the classical alpha-stable tail exponent:
    #   score = p*sgn(R)*|R|^(p-1),  q-moment iff q*(p-1)<alpha
    #   exponent grad = p'(u)*|R|^p*log|R|, q-moment iff q*p<alpha
    # The latter condition concerns ONE scalar residual/exponent gradient,
    # not a necessity theorem for cancellations among sample gradients.
    def score_qmoment_finite(alpha, exponent, q):
        assert 0 < alpha < 2 and exponent > 1 and q > 0
        return q*(exponent-1) < alpha

    def exponent_qmoment_finite(alpha, exponent, q):
        assert 0 < alpha < 2 and exponent > 0 and q > 0
        return q*exponent < alpha

    assert score_qmoment_finite(1.5, 1.5, 2)
    assert not exponent_qmoment_finite(1.5, 1.5, 1)
    assert not exponent_qmoment_finite(1.5, 1.5, 2)
    assert score_qmoment_finite(1.5, 1.2, 2)
    assert exponent_qmoment_finite(1.5, 1.2, 1)
    assert not exponent_qmoment_finite(1.5, 1.2, 2)
    assert score_qmoment_finite(1.0, 1.2, 2)
    assert not exponent_qmoment_finite(1.0, 1.2, 1)

    # For finite 0<|p'(u)| and the alpha-stable tail, the q-th moment's
    # integrability reduces at infinity to integral t^beta (log t)^q dt.
    # beta=p*q-alpha-1: it converges precisely when beta<-1.
    for alpha, exponent, q in ((1.5,1.5,1),(1.5,1.2,2),
                                (1.5,1.2,1),(1.0,1.2,2)):
        beta=exponent*q-alpha-1
        assert (beta < -1) == exponent_qmoment_finite(alpha,exponent,q)
    print("PASS idealized stable-law window: fixed-p residual-score q(p-1)<alpha")
    print("     differs from scalar learnable-exponent gradient q*p<alpha.")
    print("IMPORTANT: actual max-abs normalized residuals are bounded for")
    print("     fixed finite network parameters. No divergence is asserted.")

    print("PASS: all four budget means lower than fixed p=1.2.")
    print("These seed comparisons are conditional on one pretrained encoder")
    print("per loss and do not evaluate independently learned p trajectories.")


if __name__ == "__main__":
    audit()
