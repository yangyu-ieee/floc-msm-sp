"""Recompute the compact R3.1 local-slice summary from all archived curves.

This checks arithmetic/provenance only. It does not establish global geometry,
convexity, Hessian properties, or behavior outside the sampled directions/radii.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_CANDIDATES = (
    ROOT / "logs" / "r1_loss_landscape_multidir_v2" / "results.json",
    ROOT / "results" / "diagnostics" / "local_geometry" / "results.json",
)
RESULTS = next((path for path in RESULT_CANDIDATES if path.exists()), RESULT_CANDIDATES[0])
MAIN = ROOT / "manuscript" / "sp_main.tex"
SUPPLEMENT_CANDIDATES = (
    ROOT / "manuscript" / "sp_supplementary.tex",
    ROOT / "supplement" / "sp_supplementary.tex",
)
SUPPLEMENT = next((path for path in SUPPLEMENT_CANDIDATES if path.exists()), SUPPLEMENT_CANDIDATES[0])


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    curves = data["local_loss_landscape"]
    radii = next(iter(curves.values()))["relative_radii"]
    center_index = radii.index(0.0)
    left, right, endpoint_average = [], [], []
    curve_count = 0
    for item in curves.values():
        for curve in item["direction_curves"]:
            values = curve["heldout_objective"]
            if len(values) != len(radii) or not all(math.isfinite(v) and v > 0 for v in values):
                raise AssertionError("Malformed/nonpositive held-out local objective curve")
            center = values[center_index]
            left_change = 100.0 * (values[0] / center - 1.0)
            right_change = 100.0 * (values[-1] / center - 1.0)
            left.append(left_change)
            right.append(right_change)
            endpoint_average.append((left_change + right_change) / 2.0)
            curve_count += 1

    if curve_count != 60:
        raise AssertionError(f"Expected 60 local slices; got {curve_count}")
    if not all(x > 0 for x in left + right):
        raise AssertionError("At least one endpoint did not exceed its trained-center objective")

    summary = {
        "n_local_slices": curve_count,
        "left_endpoint_percent": {
            "median": statistics.median(left), "min": min(left), "max": max(left)
        },
        "right_endpoint_percent": {
            "median": statistics.median(right), "min": min(right), "max": max(right)
        },
        "mean_of_two_endpoint_changes_percent": {
            "median": statistics.median(endpoint_average),
            "min": min(endpoint_average),
            "max": max(endpoint_average),
        },
        "result_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
    }

    expected_summary = {
        "left_endpoint_median": 3.29,
        "right_endpoint_median": 3.06,
        "two_endpoint_median": 3.15,
        "two_endpoint_min": 2.10,
        "two_endpoint_max": 4.89,
    }
    observed_summary = {
        "left_endpoint_median": summary["left_endpoint_percent"]["median"],
        "right_endpoint_median": summary["right_endpoint_percent"]["median"],
        "two_endpoint_median": summary["mean_of_two_endpoint_changes_percent"]["median"],
        "two_endpoint_min": summary["mean_of_two_endpoint_changes_percent"]["min"],
        "two_endpoint_max": summary["mean_of_two_endpoint_changes_percent"]["max"],
    }
    for key, expected in expected_summary.items():
        if round(observed_summary[key], 2) != expected:
            raise AssertionError(f"{key} mismatch: {observed_summary[key]} != {expected}")

    text = MAIN.read_text(encoding="utf-8")
    if "median endpoint increase 3.15\\%, range 2.10--4.89\\%" not in text:
        raise AssertionError("Main text lacks its current audited local-slice summary")
    supp = SUPPLEMENT.read_text(encoding="utf-8")
    label = r"\label{fig:supp_local_landscape}"
    if label not in supp:
        raise AssertionError("Supplementary local-slice figure is missing")
    end = supp.index(label)
    start = supp.rfind(r"\begin{figure}", 0, end)
    caption = supp[start:end]
    if "median two-endpoint increase is 3.15\\%" not in caption or "2.10--4.89\\%" not in caption:
        raise AssertionError("Supplementary local-slice caption lacks the audited summary")

    output = RESULTS.with_name("local_landscape_endpoint_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("PASS: 60/60 local slices; endpoint medians 3.29% (negative), 3.06% (positive)")
    print("PASS: two-endpoint average median 3.15%, range 2.10--4.89%")
    print(f"Source SHA-256: {summary['result_sha256']}")
    print("Scope: local sampled geometry only; not global/Hessian evidence")


if __name__ == "__main__":
    main()
