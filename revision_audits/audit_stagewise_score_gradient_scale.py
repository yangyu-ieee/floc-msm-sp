"""Data-free audit of stagewise score concentration versus raw parameter-gradient scale.

Run from repository root:
    python revision_audits/audit_stagewise_score_gradient_scale.py

Uses only Python standard library and archived v70 JSON and source text.
No model training. These descriptive differences are NOT formal independent tests.
"""
from __future__ import annotations

import json
from math import isclose
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
STAGE = BASE / "results/diagnostics/stagewise"
ACTUAL = BASE / "results/diagnostics/actual_residuals"
EPOCHS = (1, 5, 10, 20)
LOSSES = ("mse", "huber", "floc_1.2", "charbonnier", "l1")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(items):
    return sum(items) / len(items)


def grouped(records, loss, epoch=None):
    out = [r for r in records if r["loss"] == loss
           and (epoch is None or r["epoch"] == epoch)]
    out.sort(key=lambda x: x["seed"])
    assert len(out) == 5, (loss, epoch, len(out))
    assert [r["seed"] for r in out] == list(range(5))
    return out


def check() -> None:
    stage = load(STAGE / "results.json")["records"]
    published = load(STAGE / "stagewise_gradient_tail_summary.json")["summary"]
    actual = load(ACTUAL / "results.json")["records"]
    assert len(stage) == 100 and len(actual) == 25
    assert all(r["n_residuals"] == 128000 for r in stage + actual)
    assert all(r["residual"]["abs_q99"] < 1 for r in stage + actual)
    assert all(len(r["parameter_gradient_norm"]["batch_values"]) == 8 for r in stage)

    diagnostics = (BASE / "code/r1_stagewise_diagnostics.py").read_text(encoding="utf-8")
    base = (BASE / "code/r1_actual_residual_diagnostics.py").read_text(encoding="utf-8")
    assert "enc.eval()" in diagnostics and "dec.eval()" in diagnostics
    assert "enc.train()" in diagnostics and "dec.train()" in diagnostics
    assert "torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)" in diagnostics
    assert "param_grad_norms.append(float(np.sqrt(grad_sq)))" in diagnostics
    assert "list(enc.parameters()) + list(dec.parameters()) + [mask_tok]" in diagnostics
    assert "if name == \"mse\":" in base and "return 2.0 * r" in base
    assert "if name == \"huber\":" in base and "return np.clip(r, -1.0, 1.0)" in base
    assert "k = max(1, int(np.ceil(0.001 * len(sq))))" in base
    assert "if name == \"l1\":" in base and "return np.sign(r)" in base

    higher = 0
    lower_score = 0
    huber_grad_lower = 0
    print("epoch | MSE q90 | p=1.2 q90 | p/MSE | p-score share / MSE")
    for epoch in EPOCHS:
        mse = grouped(stage, "mse", epoch)
        p12 = grouped(stage, "floc_1.2", epoch)
        huber = grouped(stage, "huber", epoch)
        def q90(rows):
            return [r["parameter_gradient_norm"]["q90"] for r in rows]
        def share(rows):
            return [r["score"]["top_0p1pct_squared_score_share"] for r in rows]
        mgrad, pgrad, hgrad = q90(mse), q90(p12), q90(huber)
        ms, ps = share(mse), share(p12)
        higher += sum(a > b for a, b in zip(pgrad, mgrad))
        lower_score += sum(a < b for a, b in zip(ps, ms))
        huber_grad_lower += sum(a < b for a, b in zip(hgrad, mgrad))
        assert isclose(mean(mgrad), published[str(epoch)]["mse"]["q90"]["mean"], abs_tol=1e-12)
        assert isclose(mean(pgrad), published[str(epoch)]["floc_1.2"]["q90"]["mean"], abs_tol=1e-12)
        print(f"{epoch:>5} | {mean(mgrad):.6f} | {mean(pgrad):.6f} | "
              f"{mean(pgrad)/mean(mgrad):.3f} | {mean(ps)/mean(ms):.3f}")

    assert (higher, lower_score, huber_grad_lower) == (19, 20, 20)
    print("PASS: 19/20 p=1.2 seed-epoch gradients q90 above MSE,")
    print("      20/20 score shares below MSE; 20/20 Huber q90 below MSE.")

    l1 = [r["score"]["top_0p1pct_squared_score_share"]
          for r in stage if r["loss"] == "l1"]
    assert len(l1) == 20 and all(isclose(s, .001, abs_tol=1e-12) for s in l1)
    l1actual = [r["score"]["top_0p1pct_squared_score_share"]
                for r in actual if r["loss"] == "l1"]
    assert len(l1actual) == 5 and all(isclose(s, .001, abs_tol=1e-12)
                                      for s in l1actual)
    print("PASS: all 25 L1 snapshots equal the 0.1% top-score-share floor.")
    print("      This is expected from equal-magnitude nonzero L1 scores,")
    print("      not a network-learning achievement.")

    e20m = grouped(stage, "mse", 20)
    e20h = grouped(stage, "huber", 20)
    ratio = mean([r["parameter_gradient_norm"]["q90"] for r in e20h]) / mean(
        [r["parameter_gradient_norm"]["q90"] for r in e20m])
    assert isclose(ratio, 0.49296037693701233, abs_tol=1e-10)
    print(f"Huber q90/MSE q90 at epoch 20: {ratio:.6f};")
    print("on abs(residual)<=1, Huber delta=1 has half the MSE gradient")
    print("under the archived MSE=r**2, Huber=0.5*r**2 conventions.")
    print("This scaling relation does not identify the full cross-model mechanism.")
    print("Held-out gradients are recomputed with dropout OFF and unclipped,")
    print("on 8 batches per seed; training uses dropout ON and encoder-only clipping.")


if __name__ == "__main__":
    check()
