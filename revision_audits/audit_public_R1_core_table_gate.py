"""Public-release-only gate for two tables in the clean, assembled R1 PDF.

Run from repository root:
    python revision_audits/audit_public_R1_core_table_gate.py

No outside data, Torch, SciPy, GPU, supplementary TeX, or network needed.
Checks 16 Table 1 and 12 Table 3 mean/SD printed entries directly against
the frozen v70 archived seed-level JSON. Also checks source coverage status
without claiming that unpublished supplementary tables were independently
audited. The printed constants were transcribed from clean R1 PDF pages
78 and 84 of SIGPRO-D-26-02622_R1.pdf and checked against the archive
in an independent connected-GitHub numeric comparison.
"""
from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "revision_v70_20261010"
BUDGETS = ("50", "100", "200", "500")
# Each cell is (mean%, population SD%) as printed to ONE decimal.
TABLE_1 = {
    "mse": [("93.6","1.3"), ("94.5","1.5"), ("95.7","1.9"), ("97.0","1.0")],
    "l1": [("93.7","1.7"), ("96.8","1.3"), ("96.8","1.6"), ("98.6","0.7")],
    "huber": [("92.5","0.6"), ("94.2","1.1"), ("96.2","0.9"), ("97.4","1.0")],
    "floc_1.2": [("94.1","0.7"), ("96.0","1.1"), ("97.0","0.9"), ("98.5","0.4")],
}
TABLE_3 = {
    "huber": [("94.2","1.5"), ("96.7","0.7"), ("97.0","0.7"), ("98.4","0.3")],
    "charbonnier": [("95.3","0.8"), ("96.9","1.1"), ("98.0","0.3"), ("98.0","0.3")],
    "mcc": [("93.9","0.7"), ("96.6","1.0"), ("97.8","0.8"), ("97.6","0.9")],
}


def read(relative: str):
    return json.loads((BASE / relative).read_text(encoding="utf-8"))


def rounded_pct(value: float) -> Decimal:
    return Decimal(str(value*100)).quantize(Decimal("0.1"),
                                          rounding=ROUND_HALF_UP)


def verify_record(record, expected, label):
    seeds = record["seeds"]
    assert len(seeds) == 5, label
    mean = sum(seeds)/5
    sd = (sum((x-mean)**2 for x in seeds)/5)**0.5
    assert isclose(mean, record["mean"], abs_tol=5e-7), label
    assert isclose(sd, record["std"], abs_tol=5e-7), label
    displayed = rounded_pct(record["mean"]),rounded_pct(record["std"])
    wanted = Decimal(expected[0]),Decimal(expected[1])
    assert displayed == wanted, (label, displayed, wanted)


def test_table_one():
    data = read("results/main_table_reconciliation_20261009/results.json")
    info = read("results/main_table_reconciliation_20261009/run_metadata.json")
    assert "superseded for manuscript Table 1" in info["notes"]
    total = 0
    for name, row in TABLE_1.items():
        for budget,shown in zip(BUDGETS,row):
            verify_record(data[name][budget], shown, f"Table 1/{name}/{budget}")
            total += 1
    assert total == 16
    print("PASS clean R1 Table 1: 16/16 printed mean/SD cells, one")
    print("     explicitly designated corrected reconciliation archive.")


def test_table_three():
    selections = read("results/validation_selected/validation_selection.json")
    selected_100 = read(
        "results/validation_selected_n100/validation_selection.json")
    assert "100" not in selections
    assert "100" in selected_100
    selections.update(selected_100)
    total = 0
    for family, row in TABLE_3.items():
        for budget,shown in zip(BUDGETS,row):
            s = selections[budget][family]
            assert s["selected_variant"] in s["all_candidate_validation_means"]
            # The original source picks by mean validation accuracy, not TEST.
            assert isclose(
                s["validation_mean"],
                max(s["all_candidate_validation_means"].values()),
                abs_tol=5e-7)
            reconstructed = {
                "seeds":s["heldout_test_seeds"],
                "mean":s["heldout_test_mean"],
                "std":s["heldout_test_std"],
            }
            verify_record(reconstructed, shown, f"Table 3/{family}/{budget}")
            total += 1
    assert total == 12
    print("PASS clean R1 Table 3: 12/12 printed mean/SD cells,")
    print("     selection matches validation-only candidate grid.")


def preflight_supplement_checker():
    # Frozen v70 deliberately does not redistribute full manuscript source.
    # The source-dependent audits must not be represented as passing here.
    missing = (
        "manuscript/sp_main.tex",
        "manuscript/ucr_cms_dataset_table.tex",
        "manuscript/noisex92_dataset_table.tex",
        "SOURCE_PROVENANCE.json",
        "results/diagnostics/epoch20_score_samples.npz",
    )
    missing_actual = [path for path in missing if not (BASE/path).is_file()]
    supp_alternatives = [
        "manuscript/sp_supplementary.tex",
        "supplement/sp_supplementary.tex",
    ]
    supp_available = any((BASE/path).is_file() for path in supp_alternatives)
    print(f"Source dependency preflight: {len(missing_actual)} of "
          f"{len(missing)} inputs absent; supplement TeX present={supp_available}.")
    if missing_actual or not supp_available:
        print("SKIP full audit_supplement_tables.py and audit_release.py")
        print("     in this public release: missing TeX/provenance/raw arrays.")
        for path in missing_actual:
            print(f"     absent: {path}")
    else:
        print("PRESENT source prerequisites; full supplementary audit")
        print("     remains UNEXECUTED by this standalone gate.")
    print("This tool verifies ONLY Table 1 and Table 3 printed cells,")
    print("not the entirety of Supplementary Tables S1-S18 or full R1.")


def main():
    test_table_one()
    test_table_three()
    preflight_supplement_checker()
    print("PASS: public-source core-table evidence gate, limited scope.")


if __name__ == "__main__":
    main()
