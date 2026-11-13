"""Audit 26SP R1 Figure S1 local slices without retraining.

Requires only the Python standard library. It reads the frozen raw 60 curves,
archived compact summary, config and runner source. It neither edits v70 nor
uses a GPU.

Run: python revision_audits/audit_local_landscape_fixed_slices.py
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "revision_v70_20261010"
CODE = ROOT / "code/r1_loss_landscape.py"
DIR = ROOT / "results/diagnostics/local_geometry"
LOSSES = ("mse", "l1", "charbonnier", "floc_1.2")
SEEDS = range(3)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> None:
    cfg = read_json(DIR / "config.json")
    raw_bytes = (DIR / "results.json").read_bytes()
    data = json.loads(raw_bytes)
    compact = read_json(DIR / "local_landscape_endpoint_summary.json")
    src = CODE.read_text(encoding="utf-8")
    assert cfg["script_sha256"] == hashlib.sha256(CODE.read_bytes()).hexdigest()
    assert cfg["losses"] == list(LOSSES) and cfg["seeds"] == 3
    assert cfg["random_directions_per_trained_model"] == 5
    assert cfg["validation_samples"] == 1000
    assert compact["result_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert "enc.eval(); dec.eval()" in src
    assert "x_val[:min(128, len(x_val))]" in src
    assert "fixed_mask = torch.rand(" in src
    assert "generator=mask_gen" in src
    assert "p.copy_(p0 + radius * parameter_norm * d)" in src
    assert "direction = [d / (direction_norm + 1e-12) for d in direction]" in src
    assert "direction_gen = torch.Generator(device=device).manual_seed(78000 + seed * 100 + direction_id)" in src
    assert "for p, p0 in zip(params, base):" in src
    assert "p.copy_(p0)" in src

    curves = data["local_loss_landscape"]
    expected = {f"seed={seed}/loss={loss}" for seed in SEEDS for loss in LOSSES}
    assert set(curves) == expected
    mid_inc, endpoint_inc, ratios = [], [], []
    left, right, center_values = [], [], []
    strict_count = 0
    for key, item in curves.items():
        assert item["relative_radii"] == [-0.1, -0.05, 0.0, 0.05, 0.1]
        assert item["validation_examples"] == 128
        seed = int(key.split("/")[0].split("=")[1])
        assert item["fixed_mask_seed"] == 79000 + seed
        assert len(item["direction_curves"]) == 5
        assert {c["direction_id"] for c in item["direction_curves"]} == set(range(5))
        values = [c["heldout_objective"] for c in item["direction_curves"]]
        c0 = values[0][2]
        assert all(v[2] == c0 for v in values), key
        center_values.append(c0)
        for v in values:
            assert len(v) == 5
            assert all(math.isfinite(a) and a > 0 for a in v)
            # Stronger than the archived endpoint-only check:
            assert v[0] > v[1] > v[2] < v[3] < v[4]
            strict_count += 1
            inner = 100*((v[1]+v[3])/(2*v[2])-1)
            outer = 100*((v[0]+v[4])/(2*v[2])-1)
            assert inner > 0 and outer > inner
            mid_inc.append(inner)
            endpoint_inc.append(outer)
            ratios.append(outer/inner)
            left.append(100*(v[0]/v[2]-1))
            right.append(100*(v[4]/v[2]-1))

    assert strict_count == 60 and len(center_values) == 12
    assert math.isclose(statistics.median(endpoint_inc),
                        compact["mean_of_two_endpoint_changes_percent"]["median"],
                        abs_tol=1e-9)
    assert math.isclose(min(endpoint_inc),
                        compact["mean_of_two_endpoint_changes_percent"]["min"],
                        abs_tol=1e-9)
    assert math.isclose(max(endpoint_inc),
                        compact["mean_of_two_endpoint_changes_percent"]["max"],
                        abs_tol=1e-9)
    assert math.isclose(statistics.median(left),
                        compact["left_endpoint_percent"]["median"], abs_tol=1e-9)
    assert math.isclose(statistics.median(right),
                        compact["right_endpoint_percent"]["median"], abs_tol=1e-9)
    assert math.isclose(statistics.median(mid_inc), 0.7977593157880047, abs_tol=1e-8)
    assert math.isclose(statistics.median(ratios), 4.011268511774332, abs_tol=1e-8)

    print("PASS: 60/60 symmetric slices strictly rise on both sides at radii 5% and 10%.")
    print("PASS: five directions on each of 12 checkpoints share exactly the center value.")
    print("PASS: fixed 128 held-out examples and fixed masks; dropout-off model evaluation.")
    print(f"Median (radius 5%) two-sided rise: {statistics.median(mid_inc):.6f}%")
    print(f"Median (radius 10%) two-sided rise: {statistics.median(endpoint_inc):.6f}%")
    print(f"Median individual outer/inner rise ratio: {statistics.median(ratios):.6f}")
    print("These 60 slices are 5 directions for each of 12 trained checkpoints,")
    print("NOT 60 independent trained models, a Hessian spectrum, or a convexity proof.")
    print("No model training or archive modification performed.")


if __name__ == "__main__":
    audit()
