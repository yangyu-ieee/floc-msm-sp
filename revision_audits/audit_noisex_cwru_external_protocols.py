"""Archived-record-only audit of 26SP NOISEX-92 and CWRU external checks.

Run from the repository root:
    python revision_audits/audit_noisex_cwru_external_protocols.py
Requires only Python standard library. No data downloads or GPU retraining.
The geometry/normalization statements in the accompanying note are
derived from the archived runner source and are not empirical estimates.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
from math import isclose

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "revision_v70_20261010" / "results"

def read(relative):
    return json.loads((BASE / relative).read_text(encoding="utf-8"))

def mean(v):
    return sum(v) / len(v)

def main():
    noise = read("noisex/results.json")
    noise_summary = read("noisex/summary.json")
    cells = sorted({name.rsplit("/loss=", 1)[0] for name in noise})
    methods = ("scratch", "mse", "l1", "floc")
    assert len(cells) == 16
    for cell in cells:
        for method in methods:
            assert len(noise[f"{cell}/loss={method}"]["seeds"]) == 5

    averages = {m: mean([noise[f"{c}/loss={m}"]["mean"] for c in cells]) for m in methods}
    expected = {
        "scratch": 0.5208433941006662,
        "mse": 0.5839172396808863,
        "l1": 0.5829730592668055,
        "floc": 0.5817487459629773,
    }
    for m in methods:
        assert isclose(averages[m], expected[m], abs_tol=1e-9)
    winners = Counter()
    for c in cells:
        winner = max(methods, key=lambda m: noise[f"{c}/loss={m}"]["mean"])
        winners[winner] += 1
    assert {m: winners[m] for m in methods} == noise_summary["win_count"]
    print("NOISEX-92 5 dB: 16 equally weighted dataset/shot conditions")
    for m in methods:
        print(f"  {m:>8}: mean accuracy {averages[m]*100:.4f}%; wins {winners[m]}/16")
    for m in ("l1", "floc"):
        better = sum(noise[f"{c}/loss={m}"]["mean"] > noise[f"{c}/loss=mse"]["mean"] for c in cells)
        tie = sum(noise[f"{c}/loss={m}"]["mean"] == noise[f"{c}/loss=mse"]["mean"] for c in cells)
        print(f"  {m} vs MSE: mean difference {(averages[m]-averages['mse'])*100:+.4f} pp; better {better}, ties {tie}")
    assert noise_summary["std_scope"] == "complete pipeline replicate; each seed re-mixes noise and re-pretrains"

    cwru = read("cwru/metadata.json")
    cwru_results = read("cwru/results.json")
    assert cwru["train_loads"] == [0, 1, 2] and cwru["test_loads"] == [3]
    assert cwru["seq_len"] == 1024 and cwru["stride"] == 512
    assert len(cwru["input_manifest"]) == 16
    assert len([x for x in cwru["input_manifest"] if "/1730_" in x["path"].replace("\\", "/")]) == 4
    assert cwru["train_shape"] == [2850, 1024] and cwru["test_shape"] == [953, 1024]
    print("\nCWRU: 12 train-load files, 4 test-load files")
    print("  2,850 train segments; 953 test segments; segment length 1,024, stride 512 (50% nominal overlap)")
    for shots in (10, 20, 50):
        key = f"shots={shots}/loss="
        top = max(("scratch", "mse", "l1", "huber", "charbonnier", "floc"),
                  key=lambda m: cwru_results[key+m]["mean"])
        print(f"  {shots}-shot: best mean {top} {cwru_results[key+top]['mean']*100:.2f}%")
    print("\nCWRU post-noise z-scoring implies |y_i| < sqrt(1024) = 32 for any finite segment,")
    print("up to floating-point accuracy: if sigma>0, sum_i y_i^2 = 1024*(sigma/(sigma+eps))^2 < 1024.")
    print("PASS: external archived-result checks; no GPU runs.")

if __name__ == "__main__":
    main()
