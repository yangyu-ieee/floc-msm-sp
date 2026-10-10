"""Explore threshold sensitivity of Hill tail-index estimates for trained residuals.

The stagewise archive stores absolute residual scores, not residual arrays. For
MSE and FLOC(p=1.2), invert their pointwise score maps to recover absolute
residuals. Estimates are descriptive only: temporal dependence and the small
number of independently trained seeds preclude treating coordinates as iid.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CANDIDATES = (
    ROOT / "logs" / "r1_stagewise_diagnostics_v2" / "epoch20_score_samples.npz",
    ROOT / "results" / "diagnostics" / "epoch20_score_samples.npz",
)
SAMPLES = next((path for path in SAMPLE_CANDIDATES if path.exists()), SAMPLE_CANDIDATES[0])
PACKAGED_LAYOUT = SAMPLES.parent == ROOT / "results" / "diagnostics"
RESULTS = (
    ROOT / "results" / "diagnostics" / "stagewise" / "results.json"
    if PACKAGED_LAYOUT
    else SAMPLES.with_name("results.json")
)
OUT = ROOT / "results" / "diagnostics" if PACKAGED_LAYOUT else ROOT / "logs" / "r1_tail_index_sensitivity"
OUTPUT_NAME = "tail_index_sensitivity.json" if PACKAGED_LAYOUT else "summary.json"
SEEDS, SIGNALS_PER_SEED, COORDS_PER_SIGNAL = 5, 1000, 128
K_VALUES = (100, 250, 500, 1000, 2000, 5000, 10000)


def hill_alpha(sample: np.ndarray, k: int) -> float:
    """Hill estimate of Pareto alpha from the largest k positive observations."""
    x = np.asarray(sample, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) <= k:
        raise ValueError(f"Need more than k={k} positive observations; got {len(x)}")
    x.sort()
    threshold = x[-k - 1]
    mean_log_excess = np.log(x[-k:] / threshold).mean()
    return float(1.0 / mean_log_excess) if mean_log_excess > 0 else float("inf")


def main() -> None:
    archive = np.load(SAMPLES)
    result_log = json.loads(RESULTS.read_text(encoding="utf-8"))
    expected = SEEDS * SIGNALS_PER_SEED * COORDS_PER_SIGNAL
    stage20 = [r for r in result_log["records"] if r["epoch"] == 20]
    if len(stage20) != SEEDS * 5:
        raise AssertionError(f"Expected {SEEDS * 5} epoch-20 records, got {len(stage20)}")
    rows = {}
    inverse_maps = (
        ("mse", "mse", lambda s: s / 2.0),
        ("floc_p1.2", "floc_1p2", lambda s: (s / 1.2) ** 5),
        ("charbonnier", "charbonnier", lambda s: 0.01 * s / np.sqrt(np.maximum(1.0 - s * s, 1e-24))),
    )
    for loss, score_key, inverse in inverse_maps:
        score = np.asarray(archive[score_key], dtype=np.float64)
        if score.size != expected:
            raise AssertionError(f"{loss}: expected {expected} scores, got {score.size}")
        if loss == "charbonnier" and np.any(score >= 1.0):
            raise AssertionError("Charbonnier score reached its saturation limit; inversion would be invalid")
        residual = inverse(score)
        by_seed = residual.reshape(SEEDS, SIGNALS_PER_SEED, COORDS_PER_SIGNAL)
        logged = [r for r in stage20 if r["loss"] == ("floc_1.2" if loss == "floc_p1.2" else loss)]
        logged.sort(key=lambda r: r["seed"])
        for seed, (residuals, record) in enumerate(zip(by_seed, logged, strict=True)):
            if record["seed"] != seed or record["n_residuals"] != SIGNALS_PER_SEED * COORDS_PER_SIGNAL:
                raise AssertionError(f"Unexpected seed layout for {loss}, seed={seed}")
            q99 = float(np.quantile(residuals, 0.99))
            if not np.isclose(q99, record["residual"]["abs_q99"], rtol=2e-3, atol=1e-6):
                raise AssertionError(f"Inverse score map failed q99 cross-check: {loss}, seed={seed}")
        rows[loss] = {}
        for k in K_VALUES:
            estimates = [hill_alpha(seed_residual.ravel(), k) for seed_residual in by_seed]
            rows[loss][str(k)] = {
                "k_exceedances": k,
                "tail_fraction": k / (SIGNALS_PER_SEED * COORDS_PER_SIGNAL),
                "alpha_hat_by_seed": estimates,
                "median": float(np.median(estimates)),
                "min": float(np.min(estimates)),
                "max": float(np.max(estimates)),
            }
    result = {
        "scope": "Exploratory threshold-sensitivity diagnostic; not an estimate of a stable-law alpha or a test of population moments.",
        "method": "Hill estimator on absolute residuals reconstructed by inverting the archived pointwise absolute score map.",
        "dependence_caveat": "Residual coordinates are dependent within sequences and seeds; seed ranges are descriptive, not iid confidence intervals.",
        "source": str(SAMPLES.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SAMPLES.read_bytes()).hexdigest(),
        "crosscheck": "Reconstructed per-seed q99 matches the independently logged epoch-20 residual q99 within 0.2% relative tolerance.",
        "seed_layout": {
            "independent_training_seeds": SEEDS,
            "validation_signals_per_seed": SIGNALS_PER_SEED,
            "coordinates_per_signal": COORDS_PER_SIGNAL,
        },
        "losses": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / OUTPUT_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
