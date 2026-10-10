"""Audit selected, self-contained provenance and numeric claims in this release.

This is deliberately not a full paper-claim audit. It checks packaged source
hashes where run metadata records them, the primary alpha=1.5 table, and the
recorded CWRU input hashes. Missing metadata is reported as a limitation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RESULTS = "results/main_table_reconciliation_20261009"


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded(value: float, places: int = 3) -> Decimal:
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def check_source_hash(metadata_path: str, field: str, source_path: str) -> str | None:
    metadata = read_json(metadata_path)
    expected = metadata.get(field)
    if not expected:
        return f"{metadata_path}: missing {field}"
    actual = sha256(ROOT / source_path)
    if actual != expected:
        return f"{metadata_path}: {source_path} hash mismatch ({actual} != {expected})"
    return None


def primary_table_audit() -> tuple[int, list[str]]:
    tex = (ROOT / "manuscript/sp_main.tex").read_text(encoding="utf-8")
    match = re.search(r"\\label\{tab:main_alpha15\}(.*?)\\end\{tabular\}", tex, re.S)
    if not match:
        return 0, ["main alpha=1.5 table label/body not found"]

    summary = read_json(f"{PRIMARY_RESULTS}/summary.json")
    aliases = {
        "MSE": "mse",
        "L1": "l1",
        r"Huber \delta=1": "huber",
        "p-moment loss (p=1.2)": "floc_1.2",
    }
    rows = 0
    checked = 0
    failures: list[str] = []
    for line in match.group(1).splitlines():
        row = re.match(r"^(MSE|L1|Huber .*?|\$p\$-moment loss .*?)\s*&\s*(.*?)\\\\$", line.strip())
        if not row:
            continue
        label, cell_text = row.groups()
        key = aliases.get(label.replace("$", ""))
        cell_text = cell_text.replace("$", "").replace(r"\mathbf{", "").replace("}", "")
        cells = re.findall(r"([0-9]+\.[0-9])\s*\\pm\s*([0-9]+\.[0-9])", cell_text)
        if key is None or len(cells) != 4:
            failures.append(f"unparsed table row: {line.strip()}")
            continue
        rows += 1
        for budget, (shown_mean, shown_std) in zip(("50", "100", "200", "500"), cells):
            data = summary[key][budget]
            expected = (rounded(data["mean"] * 100, 1), rounded(data["std"] * 100, 1))
            shown = (Decimal(shown_mean), Decimal(shown_std))
            checked += 1
            if shown != expected:
                failures.append(f"{label}, n={budget}: manuscript {shown}, results {expected}")
            if len(data.get("seeds", [])) != 5:
                failures.append(f"{label}, n={budget}: expected 5 seeds")
    if rows != 4 or checked != 16:
        failures.append(f"expected 4 rows/16 cells, found {rows}/{checked}")
    return checked, failures


def cwru_audit() -> tuple[int, list[str]]:
    metadata = read_json("results/cwru/metadata.json")
    results = read_json("results/cwru/results.json")
    tex = (ROOT / "supplement/sp_supplementary.tex").read_text(encoding="utf-8")
    match = re.search(r"\\label\{tab:supp_cwru\}(.*?)\\end\{tabular\}", tex, re.S)
    failures: list[str] = []
    if not match:
        return 0, ["CWRU table label/body not found"]
    if len(metadata.get("input_manifest", [])) != 16:
        failures.append("expected 16 recorded CWRU input hashes")
    source_map = read_json("SOURCE_PROVENANCE.json")
    cwru_source = source_map["portable_adaptations"]["code/cwru_loss_compare_sourceclosed.py"]
    if metadata.get("script_sha256", "").lower() != cwru_source["historical_run_source_sha256"].lower():
        failures.append("CWRU recorded historical run source hash does not match provenance map")
    if sha256(ROOT / "code/cwru_loss_compare_sourceclosed.py") != cwru_source["packaged_portable_source_sha256"].lower():
        failures.append("CWRU packaged portable source hash does not match provenance map")

    aliases = {"Scratch": "scratch", "MSE": "mse", "L1": "l1", "Huber": "huber",
               "Charbonnier": "charbonnier", "p-moment loss (p=1.2)": "floc"}
    rows = 0
    cells_checked = 0
    for line in match.group(1).splitlines():
        row = re.match(r"^(Scratch|MSE|L1|Huber|Charbonnier|\$p\$-moment loss .*?)\s*&\s*(.*?)\\\\$", line.strip())
        if not row:
            continue
        label, cell_text = row.groups()
        label = label.replace("$", "")
        cell_text = cell_text.replace("$", "").replace(r"\mathbf{", "").replace("}", "")
        cells = re.findall(r"([0-9]+\.[0-9])\s*\\pm\s*([0-9]+\.[0-9])", cell_text)
        loss = aliases[label.replace("$", "")]
        if len(cells) != 3:
            failures.append(f"{label}: expected 3 mean/SD cells")
            continue
        rows += 1
        for shots, (mean, std) in zip((10, 20, 50), cells):
            entry = results[f"shots={shots}/loss={loss}"]
            shown = (Decimal(mean), Decimal(std))
            expected = (rounded(entry["mean"] * 100, 1), rounded(entry["std"] * 100, 1))
            cells_checked += 1
            if shown != expected:
                failures.append(f"{label}, shots={shots}: manuscript {shown}, results {expected}")
            if len(entry.get("seeds", [])) != 5:
                failures.append(f"{label}, shots={shots}: expected 5 seeds")
    if rows != 6 or cells_checked != 18:
        failures.append(f"expected 6 rows/18 cells, found {rows}/{cells_checked}")
    return cells_checked, failures


def exact_sign_flip_p(a: list[float], b: list[float]) -> float:
    differences = [x - y for x, y in zip(a, b)]
    observed = abs(sum(differences))
    assignments = itertools.product((-1, 1), repeat=len(differences))
    extreme = sum(
        abs(sum(sign * difference for sign, difference in zip(signs, differences)))
        >= observed - 1e-12
        for signs in assignments
    )
    return extreme / (2 ** len(differences))


def main_significance_audit() -> tuple[int, list[str]]:
    raw = read_json(f"{PRIMARY_RESULTS}/results.json")
    supp = (ROOT / "supplement/sp_supplementary.tex").read_text(encoding="utf-8")
    match = re.search(
        r"\\label\{tab:supp_main_significance\}(.*?)\\end\{tabular\}",
        supp,
        re.S,
    )
    if not match:
        return 0, ["main benchmark significance table not found"]
    baselines = {
        "p-moment loss vs MSE": "mse",
        "p-moment loss vs Huber": "huber",
        "p-moment loss vs L1": "l1",
    }
    failures: list[str] = []
    checked = 0
    for line in match.group(1).splitlines():
        row = re.match(r"^(p-moment loss vs MSE|p-moment loss vs Huber|p-moment loss vs L1)\s*&\s*(.*?)\\\\$", line.strip())
        if not row:
            continue
        label, values = row.groups()
        shown = [Decimal(x) for x in re.findall(r"[0-9]+\.[0-9]{3}", values)]
        if len(shown) != 4:
            failures.append(f"{label}: expected four p-values")
            continue
        baseline = baselines[label]
        for budget, reported in zip(("50", "100", "200", "500"), shown):
            calculated = rounded(
                exact_sign_flip_p(
                    raw["floc_1.2"][budget]["seeds"],
                    raw[baseline][budget]["seeds"],
                ),
                3,
            )
            checked += 1
            if reported != calculated:
                failures.append(
                    f"{label}, n={budget}: Supplement {reported}, exact test {calculated}"
                )
    if checked != 12:
        failures.append(f"expected 12 exact sign-flip p-values, found {checked}")
    return checked, failures


def main() -> int:
    failures: list[str] = []
    source_checks = (
        (f"{PRIMARY_RESULTS}/config.json", "code_sha256", "code/floc_msm_main_corrected.py"),
        (f"{PRIMARY_RESULTS}/run_metadata.json", "wrapper_script_sha256", "code/provenance/run_main_table_reconciliation_executed_20261009.py"),
    )
    for metadata, field, source in source_checks:
        error = check_source_hash(metadata, field, source)
        if error:
            failures.append(error)

    main_cells, main_failures = primary_table_audit()
    main_pvalues, main_p_failures = main_significance_audit()
    cwru_cells, cwru_failures = cwru_audit()
    failures.extend(main_failures)
    failures.extend(main_p_failures)
    failures.extend(cwru_failures)

    source_map = read_json("SOURCE_PROVENANCE.json")
    for source, details in source_map["portable_adaptations"].items():
        if sha256(ROOT / source) != details["packaged_portable_source_sha256"].lower():
            failures.append(f"{source}: portable source hash does not match provenance map")
        metadata = read_json(details["run_metadata"])
        if metadata.get(details["metadata_hash_field"], "").lower() != details["historical_run_source_sha256"].lower():
            failures.append(f"{source}: historical source hash does not match run metadata")

    print(f"source-hash checks: {len(source_checks)} exact + {len(source_map['portable_adaptations'])} documented portable adaptations")
    print(f"main alpha=1.5 manuscript cells checked: {main_cells}/16")
    print(f"main alpha=1.5 exact paired p-values checked: {main_pvalues}/12")
    print(f"CWRU manuscript cells checked: {cwru_cells}/18")
    print("CWRU input hashes recorded: 16/16 (record presence; raw data are not distributed)")
    print("Not checked: all supplementary tables, NOISEX/UCR table transcription, cross-protocol equivalence inference, full-paper claim audit")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: selected release checks only; see scope limitations above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
