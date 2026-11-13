"""Data-free audit of stagewise score concentration versus raw parameter-gradient scale.

Run from repository root:
    python revision_audits/audit_stagewise_score_gradient_scale.py

Uses only Python standard library and archived v70 JSON and source text.
No model training. These descriptive differences are NOT formal independent tests.
"""
from __future__ import annotations

import json
from math import isclose, sqrt
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
    # Evaluate the two loss score functions at the SAME hypothetical r:
    #   MSE: 2*r; p=1.2: 1.2*sign(r)*abs(r)**0.2.
    # Ratio is 0.6*abs(r)**(-0.8), >1 for abs(r)<0.6**1.25.
    # The model-specific residual quantiles bound the fraction in either
    # regime; they do NOT prove a summed parameter-gradient ordering.
    crossover = 0.6 ** (1.0 / 0.8)
    assert isclose(crossover, 0.528067042076036, abs_tol=1e-12)
    p_rows = grouped(actual, "floc_1.2")
    assert all(r["residual"]["abs_q90"] < crossover
               < r["residual"]["abs_q99"] for r in p_rows)
    ratio = lambda magnitude: 0.6 * magnitude ** (-0.8)
    median_ratios = [ratio(r["residual"]["abs_q50"]) for r in p_rows]
    q90_ratios = [ratio(r["residual"]["abs_q90"]) for r in p_rows]
    q99_ratios = [ratio(r["residual"]["abs_q99"]) for r in p_rows]
    assert 4.73 < min(median_ratios) <= max(median_ratios) < 4.88
    assert 1.47 < min(q90_ratios) <= max(q90_ratios) < 1.54
    assert 0.70 < min(q99_ratios) <= max(q99_ratios) < 0.73
    print(f"p=1.2 versus MSE residual-score crossover: |r|={crossover:.6f}")
    print(f"p=1.2 same-residual score ratios at its q50/q90/q99 "
          f"(five-run means): {mean(median_ratios):.3f}, "
          f"{mean(q90_ratios):.3f}, {mean(q99_ratios):.3f}")
    print("PASS: every p=1.2 model has q90 below, q99 above the crossing;")
    print("      about 90% or more of its 128,000 held-out coordinates have")
    print("      a *larger* p=1.2 than MSE score at the same nonzero residual.")

    # Charbonnier smooths L1 only close to r=0, for the chosen eps=0.01.
    # Its score magnitude is |r|/sqrt(r**2 + eps**2). Since this is
    # monotone, the per-run q50/q90 score quantiles can be checked directly.
    charb = grouped(actual, "charbonnier")
    for record in charb:
        for r_quantile, s_quantile in (
            ("abs_q50", "abs_q50"), ("abs_q90", "abs_q90")
        ):
            m = record["residual"][r_quantile]
            expected = m / sqrt(m*m + 0.01**2)
            assert isclose(expected, record["score"][s_quantile], abs_tol=1e-7)
    medians = [r["score"]["abs_q50"] for r in charb]
    nineties = [r["score"]["abs_q90"] for r in charb]
    assert 0.9906 < min(medians) <= max(medians) < 0.9915
    assert 0.99949 < min(nineties) <= max(nineties) < 0.99955
    print(f"Charbonnier eps=0.01 median |score| mean={mean(medians):.6f},")
    print(f"     q90 mean={mean(nineties):.6f} (L1 nonzero score=1).")
    print("PASS: at least half of held-out coordinates have Charbonnier")
    print("      score magnitude >=~0.991, but no inference is made about")
    print("      the lower half or the exact fraction within abs(r)<=0.01.")
    print("NOTE: all comparisons are score-shape calculations on the same")
    print("      residual, not controlled gradient comparisons of trained models.")

    print("Held-out gradients are recomputed with dropout OFF and unclipped,")
    print("on 8 batches per seed; training uses dropout ON and encoder-only clipping.")


if __name__ == "__main__":
    check()
