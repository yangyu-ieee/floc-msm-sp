"""Cross-check selected supplementary tables and linked claims against raw run JSON.

This is a targeted local numerical audit, not the required independent
zero-context whole-paper review.
"""

import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from scipy.stats import ttest_ind


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "manuscript" / "sp_main.tex"
SUPP_PATH = next(
    path
    for path in (
        ROOT / "manuscript" / "sp_supplementary.tex",
        ROOT / "supplement" / "sp_supplementary.tex",
    )
    if path.exists()
)
MAIN = MAIN_PATH.read_text(encoding="utf-8")
SUPP = SUPP_PATH.read_text(encoding="utf-8")


def artifact_path(local_path, packaged_path=None):
    candidates = (local_path, packaged_path) if packaged_path else (local_path,)
    for relative_path in candidates:
        if relative_path and (ROOT / relative_path).exists():
            return ROOT / relative_path
    raise FileNotFoundError(f"No audit input found among: {candidates}")


def read_json(relative_path, packaged_path=None):
    return json.loads(artifact_path(relative_path, packaged_path).read_text(encoding="utf-8"))


def table_body(source, label):
    match = re.search(
        rf"\\label\{{{re.escape(label)}\}}(.*?)\\end\{{tabular\}}",
        source,
        flags=re.S,
    )
    if not match:
        raise AssertionError(f"Table label not found: {label}")
    return match.group(1)


PAIR = re.compile(
    r"(?:\\mathbf\{)?(?P<mean>[0-9]+\.[0-9]+)\}?(?:\$)?"
    r"\\pm\s*(?P<std>[0-9]+\.[0-9]+)"
)


def pairs(line):
    return [(Decimal(m.group("mean")), Decimal(m.group("std"))) for m in PAIR.finditer(line)]


def rounded(value, places=1):
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def check_pair(actual, record, *, std_key="std", places=1, seed_count=5, label=""):
    expected = (
        rounded(record["mean"] * 100, places),
        rounded(record[std_key] * 100, places),
    )
    if actual != expected:
        raise AssertionError(f"{label}: table={actual}, raw={expected}")
    if len(record.get("seeds", [])) != seed_count:
        raise AssertionError(f"{label}: expected {seed_count} seed outcomes, got {len(record.get('seeds', []))}")


def check_dataset_table(source, label, result_path, packaged_result_path, losses, *, expected_rows, places=1):
    results = read_json(result_path, packaged_result_path)
    body = table_body(source, label)
    row_re = re.compile(r"^([A-Za-z0-9]+)\s*&\s*(5|10)\s*&")
    rows = 0
    cells = 0
    seen = set()
    for line in body.splitlines():
        match = row_re.match(line.strip())
        if not match:
            continue
        dataset, shots = match.groups()
        row_key = (dataset, int(shots))
        if row_key in seen:
            raise AssertionError(f"{label}: duplicate row {row_key}")
        seen.add(row_key)
        cells_in_row = pairs(line)
        if len(cells_in_row) != len(losses):
            raise AssertionError(f"{label}/{row_key}: expected {len(losses)} cells, got {len(cells_in_row)}")
        rows += 1
        for actual, loss in zip(cells_in_row, losses):
            key = f"{dataset}/shots={shots}/loss={loss}"
            check_pair(actual, results[key], places=places, label=f"{label}/{key}")
            cells += 1
    if rows != expected_rows:
        raise AssertionError(f"{label}: expected {expected_rows} rows, got {rows}")
    return rows, cells


def check_student_t():
    results = read_json("logs/floc_msm_student_t_nu1p5/results.json", "results/student_t/results.json")
    body = table_body(SUPP, "tab:supp_student_t")
    loss_names = {
        "MSE": "mse",
        "L1": "l1",
        "Huber": "huber",
        "Charbonnier": "charbonnier",
        "$p$-moment loss ($p=1.2$)": "floc_1.2",
    }
    rows = cells = 0
    for line in body.splitlines():
        stripped = line.strip()
        label = next((name for name in loss_names if stripped.startswith(name + " &")), None)
        if not label:
            continue
        row_pairs = pairs(line)
        if len(row_pairs) != 3:
            raise AssertionError(f"Student-t/{label}: expected 3 cells, got {len(row_pairs)}")
        for actual, budget in zip(row_pairs, ("50", "200", "500")):
            check_pair(actual, results[loss_names[label]][budget], label=f"Student-t/{label}/{budget}")
            cells += 1
        rows += 1
    if (rows, cells) != (5, 15):
        raise AssertionError(f"Student-t: expected 5 rows/15 cells, got {rows}/{cells}")
    return rows, cells


def check_masking_strategy():
    """Verify the random-token vs contiguous-block S5 table and its summary claim."""
    random_results = read_json(
        "logs/masking/random_token_exact/results.json",
        "results/masking/random_token_exact/results.json",
    )
    block_results = read_json(
        "logs/masking/contiguous_block_v2/results.json",
        "results/masking/contiguous_block_v2/results.json",
    )
    body = table_body(SUPP, "tab:supp_mask_strategy")
    loss_names = {
        "MSE": "mse",
        "L1": "l1",
        "Charbonnier": "charbonnier",
        "$p$-moment loss ($p=1.2$)": "floc_1.2",
    }
    sources = {"Random tokens": random_results, "Contiguous block": block_results}
    budgets = ("100", "200")
    rows = cells = 0
    random_higher = ties = 0
    deltas = []
    for line in body.splitlines():
        stripped = line.strip()
        strategy = next((name for name in sources if stripped.startswith(name + " &")), None)
        if strategy is None:
            continue
        match = re.match(r"^(Random tokens|Contiguous block)\s*&\s*(100|200)\s*&", stripped)
        if not match:
            raise AssertionError(f"Masking strategy malformed table row: {stripped}")
        strategy, budget = match.groups()
        row_pairs = pairs(line)
        if len(row_pairs) != len(loss_names):
            raise AssertionError(f"Masking strategy/{strategy}/{budget}: expected 4 cells, got {len(row_pairs)}")
        for actual, loss in zip(row_pairs, loss_names.values()):
            record = sources[strategy][loss][budget]
            check_pair(actual, record, places=2, label=f"Masking strategy/{strategy}/{loss}/{budget}")
            cells += 1
        rows += 1
    if (rows, cells) != (4, 16):
        raise AssertionError(f"Masking strategy: expected 4 rows/16 cells, got {rows}/{cells}")
    for budget in budgets:
        for loss in loss_names.values():
            delta = (random_results[loss][budget]["mean"] - block_results[loss][budget]["mean"]) * 100
            deltas.append(delta)
            displayed_delta = rounded(delta, 2)
            if displayed_delta > 0:
                random_higher += 1
            elif displayed_delta == 0:
                ties += 1
    if random_higher != 7 or ties != 1:
        raise AssertionError(f"Masking strategy win/tie count mismatch: higher={random_higher}, ties={ties}")
    if rounded(min(deltas), 2) != Decimal("0.00") or rounded(max(deltas), 2) != Decimal("1.60"):
        raise AssertionError(f"Masking strategy delta range mismatch: {min(deltas):.4f}--{max(deltas):.4f} pp")
    expected_claim = "higher mean in seven of eight"
    if expected_claim not in MAIN or "tied the remaining $p$-moment/200-label cell" not in MAIN:
        raise AssertionError("Main-text masking-strategy summary does not match the raw results")
    return rows, cells, random_higher, ties, min(deltas), max(deltas)


def check_normalization():
    results = read_json("logs/r5_preprocessing_audit/results.json", "results/normalization/results.json")
    body = table_body(MAIN, "tab:normalization_sensitivity")
    arms = {
        "Noisy max-abs": "cms_maxabs",
        "Clean global RMS": "cms_clean_global_rms",
    }
    losses = ("mse", "l1", "charbonnier", "floc_1.2")
    rows = cells = 0
    shifts = []
    for line in body.splitlines():
        match = re.match(r"^(Noisy max-abs|Clean global RMS)\s*&\s*(100|200)\s*&", line.strip())
        if not match:
            continue
        arm, budget = match.groups()
        row_pairs = pairs(line)
        if len(row_pairs) != 4:
            raise AssertionError(f"Normalization/{arm}/{budget}: expected 4 cells, got {len(row_pairs)}")
        for actual, loss in zip(row_pairs, losses):
            record = results[arms[arm]][loss][budget]
            check_pair(actual, record, std_key="std_ddof0", places=2, label=f"Normalization/{arm}/{loss}/{budget}")
            cells += 1
        rows += 1
    if (rows, cells) != (4, 16):
        raise AssertionError(f"Normalization: expected 4 rows/16 cells, got {rows}/{cells}")
    for loss in losses:
        for budget in ("100", "200"):
            a = results["cms_maxabs"][loss][budget]["mean"]
            b = results["cms_clean_global_rms"][loss][budget]["mean"]
            shifts.append(abs(a - b) * 100)
    if rounded(min(shifts), 1) != Decimal("1.9") or rounded(max(shifts), 1) != Decimal("3.6"):
        raise AssertionError(f"Normalization range claim mismatch: {min(shifts):.4f}--{max(shifts):.4f} pp")
    return rows, cells, min(shifts), max(shifts)


def check_validation_selected():
    selection = read_json(
        "logs/r2_mcc_validation_selected/validation_selection.json",
        "results/validation_selected/validation_selection.json",
    )
    selection.update(read_json(
        "logs/r2_mcc_validation_selected_n100/validation_selection.json",
        "results/validation_selected_n100/validation_selection.json",
    ))
    body = table_body(SUPP, "tab:supp_validation_baselines")
    row_specs = {
        "Huber": "huber",
        "Charbonnier": "charbonnier",
        "MCC": "mcc",
    }
    budgets = ("50", "100", "200", "500")
    rows = cells = 0
    for line in body.splitlines():
        stripped = line.strip()
        label = next((name for name in row_specs if stripped.startswith(name + " (")), None)
        if not label:
            continue
        prefix = re.search(r"\(\$[^=]+=(.*?)\$\)", stripped)
        if not prefix:
            raise AssertionError(f"Validation-selected/{label}: selected parameter list missing")
        shown_params = [v.strip() for v in prefix.group(1).split(",")]
        row_pairs = pairs(line)
        if len(shown_params) != 4 or len(row_pairs) != 4:
            raise AssertionError(f"Validation-selected/{label}: expected 4 params and 4 test cells")
        for budget, shown_param, actual in zip(budgets, shown_params, row_pairs):
            chosen = selection[budget][row_specs[label]]
            chosen_param = chosen["selected_variant"].rsplit("_", 1)[-1]
            if Decimal(shown_param) != Decimal(chosen_param):
                raise AssertionError(f"Validation-selected/{label}/{budget}: table parameter {shown_param}, selected {chosen_param}")
            candidates = chosen["all_candidate_validation_means"].values()
            if abs(chosen["validation_mean"] - max(candidates)) > 1e-8:
                raise AssertionError(f"Validation-selected/{label}/{budget}: selected variant is not a validation maximizer")
            test_record = {
                "mean": chosen["heldout_test_mean"],
                "std": chosen["heldout_test_std"],
                "seeds": chosen["heldout_test_seeds"],
            }
            check_pair(actual, test_record, label=f"Validation-selected/{label}/{budget}")
            cells += 1
        rows += 1
    if (rows, cells) != (3, 12):
        raise AssertionError(f"Validation-selected: expected 3 rows/12 cells, got {rows}/{cells}")
    return rows, cells


def check_main_validation_selected():
    """Check the compact main-text table against the same logged test results."""
    selection = read_json(
        "logs/r2_mcc_validation_selected/validation_selection.json",
        "results/validation_selected/validation_selection.json",
    )
    selection.update(read_json(
        "logs/r2_mcc_validation_selected_n100/validation_selection.json",
        "results/validation_selected_n100/validation_selection.json",
    ))
    body = table_body(MAIN, "tab:validation_selected_baselines")
    row_specs = {"Huber": "huber", "Charbonnier": "charbonnier", "MCC": "mcc"}
    budgets = ("50", "100", "200", "500")
    rows = cells = 0
    for line in body.splitlines():
        stripped = line.strip()
        label = next((name for name in row_specs if stripped.startswith(name + " &")), None)
        if not label:
            continue
        row_pairs = pairs(line)
        if len(row_pairs) != 4:
            raise AssertionError(f"Main validation-selected/{label}: expected 4 cells, got {len(row_pairs)}")
        for budget, actual in zip(budgets, row_pairs):
            chosen = selection[budget][row_specs[label]]
            record = {
                "mean": chosen["heldout_test_mean"],
                "std": chosen["heldout_test_std"],
                "seeds": chosen["heldout_test_seeds"],
            }
            check_pair(actual, record, label=f"Main validation-selected/{label}/{budget}")
            cells += 1
        rows += 1
    if (rows, cells) != (3, 12):
        raise AssertionError(f"Main validation-selected: expected 3 rows/12 cells, got {rows}/{cells}")
    return rows, cells


def check_alpha10():
    results = read_json(
        "logs/targeted_alpha10_sourceclosure_2026-10-07/results.json",
        "results/cauchy_level/results.json",
    )
    summary = read_json(
        "logs/targeted_alpha10_sourceclosure_2026-10-07/summary.json",
        "results/cauchy_level/summary.json",
    )
    body = table_body(SUPP, "tab:supp_targeted_alpha10")
    loss_names = {
        "MSE": "mse",
        "L1": "l1",
        "Charbonnier": "charbonnier",
        "$p$-moment loss ($p=1.2$)": "floc_1.2",
    }
    budgets = ("50", "100", "200", "500")
    rows = cells = 0
    for line in body.splitlines():
        stripped = line.strip()
        label = next((name for name in loss_names if stripped.startswith(name + " &")), None)
        if not label:
            continue
        row_pairs = pairs(line)
        if len(row_pairs) != 4:
            raise AssertionError(f"Alpha10/{label}: expected 4 cells, got {len(row_pairs)}")
        for budget, actual in zip(budgets, row_pairs):
            record = results[f"loss={loss_names[label]}/n={budget}"]
            check_pair(actual, record, seed_count=10, label=f"Alpha10/{label}/{budget}")
            cells += 1
        rows += 1
    if (rows, cells) != (4, 16):
        raise AssertionError(f"Alpha10: expected 4 rows/16 cells, got {rows}/{cells}")
    raw_p = [
        float(ttest_ind(
            results[f"loss=floc_1.2/n={budget}"]["seeds"],
            results[f"loss=mse/n={budget}"]["seeds"],
            equal_var=False,
        ).pvalue)
        for budget in budgets
    ]
    order = sorted(range(len(raw_p)), key=raw_p.__getitem__)
    adjusted_p = [0.0] * len(raw_p)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(raw_p) - rank) * raw_p[index])
        adjusted_p[index] = min(1.0, running_max)
    expected_raw = [1.09e-5, 4.41e-7, 2.62e-8, 2.47e-5]
    expected_holm = [2.18e-5, 1.32e-6, 1.05e-7, 2.47e-5]
    if any(abs(actual - expected) > expected * 0.01
           for actual, expected in zip(raw_p, expected_raw)):
        raise AssertionError(f"Alpha10 unpaired Welch raw p-values mismatch: {raw_p}")
    if any(abs(actual - expected) > expected * 0.01
           for actual, expected in zip(adjusted_p, expected_holm)):
        raise AssertionError(f"Alpha10 four-budget Holm p-values mismatch: {adjusted_p}")
    if "not paired by their displayed run index" not in SUPP:
        raise AssertionError("Alpha10 independent-stream limitation is missing")
    raw_latex = (
        "1.09\\times10^{-5}", "4.41\\times10^{-7}",
        "2.62\\times10^{-8}", "2.47\\times10^{-5}",
    )
    if any(value not in SUPP for value in raw_latex):
        raise AssertionError("Alpha10 raw Welch p-values are missing from Note S18")
    if "2.18\\times10^{-5}" not in SUPP or "1.32\\times10^{-6}" not in SUPP:
        raise AssertionError("Alpha10 corrected Holm values are missing from Note S18")
    return rows, cells, raw_p, adjusted_p


def check_noisex_wins():
    results = read_json(
        "logs/noisex92_seeded_rerun_2026-10-07/results.json",
        "results/noisex/results.json",
    )
    summary = read_json(
        "logs/noisex92_seeded_rerun_2026-10-07/summary.json",
        "results/noisex/summary.json",
    )
    losses = ("scratch", "mse", "l1", "floc")
    wins = {loss: 0 for loss in losses}
    conditions = sorted({key.rsplit("/loss=", 1)[0] for key in results})
    for condition in conditions:
        means = {loss: results[f"{condition}/loss={loss}"]["mean"] for loss in losses}
        winners = [loss for loss, value in means.items() if value == max(means.values())]
        if len(winners) != 1:
            raise AssertionError(f"NOISEX win-count tie in {condition}: {winners}")
        wins[winners[0]] += 1
    if len(conditions) != 16 or wins != summary["win_count"]:
        raise AssertionError(f"NOISEX raw wins disagree with run summary: {wins} vs {summary['win_count']}")
    body = table_body(SUPP, "tab:supp_noisex_wins")
    line = next((line for line in body.splitlines() if line.strip().startswith("Best-mean wins &")), None)
    counts = re.findall(r"\d+", line or "")
    if len(counts) != 4:
        raise AssertionError("NOISEX win-count row missing")
    shown = dict(zip(losses, map(int, counts)))
    if shown != wins:
        raise AssertionError(f"NOISEX table win counts disagree: {shown} vs {wins}")
    return len(conditions), wins


def check_residual_tail_sensitivity():
    summary = read_json(
        "logs/r1_tail_index_sensitivity/summary.json",
        "results/diagnostics/tail_index_sensitivity.json",
    )
    sample_candidates = (
        ROOT / "logs" / "r1_stagewise_diagnostics_v2" / "epoch20_score_samples.npz",
        ROOT / "results" / "diagnostics" / "epoch20_score_samples.npz",
    )
    sample_path = next((path for path in sample_candidates if path.exists()), None)
    if sample_path is None:
        raise FileNotFoundError("Archived epoch-20 score samples are missing")
    import hashlib
    if hashlib.sha256(sample_path.read_bytes()).hexdigest() != summary["source_sha256"]:
        raise AssertionError("Residual-tail summary source hash does not match archived score samples")
    body = table_body(SUPP, "tab:tail_index_sensitivity")
    expected_k = (100, 1000, 5000, 10000)
    loss_columns = ("mse", "charbonnier", "floc_p1.2")
    rows = cells = 0
    for line in body.splitlines():
        fields = [field.strip() for field in line.strip().rstrip("\\").split("&")]
        if not fields or not fields[0].replace(",", "").isdigit():
            continue
        k = int(fields[0].replace(",", ""))
        if k not in expected_k:
            raise AssertionError(f"Unexpected residual-tail threshold row: {k}")
        fraction = Decimal(fields[1])
        expected_fraction = rounded(summary["losses"]["mse"][str(k)]["tail_fraction"] * 100, 3)
        if fraction != expected_fraction:
            raise AssertionError(f"Tail fraction mismatch at k={k}: {fraction} vs {expected_fraction}")
        for loss, shown in zip(loss_columns, fields[2:5]):
            match = re.fullmatch(r"([0-9.]+)\s*\[([0-9.]+),\s*([0-9.]+)\]", shown)
            if not match:
                raise AssertionError(f"Malformed tail-index table cell at k={k}: {shown}")
            actual = tuple(Decimal(value) for value in match.groups())
            record = summary["losses"][loss][str(k)]
            expected = tuple(rounded(record[key], 1) for key in ("median", "min", "max"))
            if actual != expected:
                raise AssertionError(f"Tail-index table mismatch at {loss}/k={k}: {actual} vs {expected}")
            cells += 1
        rows += 1
    if (rows, cells) != (len(expected_k), len(expected_k) * len(loss_columns)):
        raise AssertionError(f"Tail sensitivity table expected 4 rows/12 cells, got {rows}/{cells}")
    if "do not exhibit a threshold-stable plateau" not in SUPP or "Supplementary Note~S14" not in MAIN:
        raise AssertionError("Residual-tail interpretation or main-text pointer is missing")
    return rows, cells, summary["source_sha256"]


def main():
    reports = {}
    reports["UCR S1"] = check_dataset_table(
        (ROOT / "manuscript" / "ucr_cms_dataset_table.tex").read_text(encoding="utf-8"),
        "tab:ucr_per_dataset",
        "logs/ucr_cms_maxabs_alpha1p5/results.json",
        "results/ucr_corrected/results.json",
        ("scratch", "mse", "l1", "huber", "charbonnier", "floc"),
        expected_rows=20,
    )
    reports["NOISEX S2"] = check_dataset_table(
        (ROOT / "manuscript" / "noisex92_dataset_table.tex").read_text(encoding="utf-8"),
        "tab:noisex_per_dataset",
        "logs/noisex92_seeded_rerun_2026-10-07/results.json",
        "results/noisex/results.json",
        ("scratch", "mse", "l1", "floc"),
        expected_rows=16,
    )
    reports["NOISEX S3"] = check_noisex_wins()
    reports["Student-t S6"] = check_student_t()
    reports["Masking strategy S5"] = check_masking_strategy()
    reports["Normalization main table"] = check_normalization()
    reports["Validation-selected S12"] = check_validation_selected()
    reports["Validation-selected main table"] = check_main_validation_selected()
    reports["Cauchy S13"] = check_alpha10()
    reports["Residual-tail S14"] = check_residual_tail_sensitivity()
    for name, result in reports.items():
        print(f"PASS {name}: {result}")
    print("PASS: selected supplementary tables and linked validation/Cauchy claims match raw results")
    print("Scope: local targeted numerical audit only; not an independent whole-paper claim audit")


if __name__ == "__main__":
    main()
