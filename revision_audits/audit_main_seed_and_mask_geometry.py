"""Non-training audit for the submitted 26SP R1 evidence.

Recomputes five-seed paired sign-flip tests and exact raw-sample
concealment probabilities implied by the Conv1D tokenizer and token masks.
This is an independent analysis script; it does not modify the v70 snapshot.
"""
from __future__ import annotations

from itertools import product
from math import comb
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "revision_v70_20261010/results/main_table_reconciliation_20261009/results.json"

def signflip_p(differences: list[float]) -> float:
    observed = abs(sum(differences))
    n = len(differences)
    extreme = sum(
        abs(sum(s * x for s, x in zip(signs, differences))) >= observed - 1e-12
        for signs in product((-1, 1), repeat=n)
    )
    return extreme / (2 ** n)

def raw_token_coverages(length=128, kernel=8, stride=4, padding=2):
    tokens = (length + 2 * padding - kernel) // stride + 1
    covers = []
    for t in range(length):
        covers.append(tuple(
            i for i in range(tokens)
            if stride * i - padding <= t < stride * i - padding + kernel
        ))
    assert all(covers), "Raw positions without a tokenizer path must be handled separately"
    return tokens, covers

def exact_random_count_hidden(tokens, covers, n_mask):
    denominator = comb(tokens, n_mask)
    return sum(
        comb(tokens - len(c), n_mask - len(c)) / denominator
        if n_mask >= len(c) else 0.0
        for c in covers
    )

def independent_bernoulli_hidden(covers, p=0.5):
    return sum(p ** len(c) for c in covers)

def contiguous_hidden_counts(tokens, covers, n_mask):
    counts = []
    for start in range(tokens - n_mask + 1):
        masked = set(range(start, start + n_mask))
        counts.append(sum(all(j in masked for j in c) for c in covers))
    return counts

def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    expectations = {"50": 0.5625, "100": 0.1875, "200": 0.1875, "500": 0.1875}
    print("Corrected alpha=1.5 comparison: exact two-sided paired sign-flip tests")
    for budget, target in expectations.items():
        a = results["floc_1.2"][budget]["seeds"]
        b = results["mse"][budget]["seeds"]
        assert len(a) == len(b) == 5
        diffs = [x - y for x, y in zip(a, b)]
        p = signflip_p(diffs)
        assert abs(p - target) < 1e-12, (budget, p, target)
        print(f"  n={budget:>3}: gain={100*sum(diffs)/len(diffs):+.2f} pp; "
              f"wins={sum(d>0 for d in diffs)}/5; p={p:.4f}")

    token_count, covers = raw_token_coverages()
    assert token_count == 32
    assert sum(len(c) == 1 for c in covers) == 4
    assert sum(len(c) == 2 for c in covers) == 124

    m = 16
    random_hidden = exact_random_count_hidden(token_count, covers, m)
    bernoulli_hidden = independent_bernoulli_hidden(covers)
    block_counts = contiguous_hidden_counts(token_count, covers, m)
    block_mean = sum(block_counts) / len(block_counts)
    assert abs(random_hidden - 32.0) < 1e-10
    assert abs(bernoulli_hidden - 33.0) < 1e-10
    assert len(block_counts) == 17 and block_counts.count(60) == 15 and block_counts.count(62) == 2
    assert abs(block_mean - 1024 / 17) < 1e-10

    # Mean accuracy is built from float32 per-seed accuracies. Equality at the
    # printed percentage-point resolution must not use bitwise float equality:
    # p=1.2 / 200 has a 1.19e-8 *fractional-accuracy* discrepancy.
    mr = json.loads((ROOT / "revision_v70_20261010/results/masking/random_token_exact/results.json").read_text(encoding="utf-8"))
    mb = json.loads((ROOT / "revision_v70_20261010/results/masking/contiguous_block_v2/results.json").read_text(encoding="utf-8"))
    comparison_tolerance = 1e-7  # accuracy fraction, 0.00001 percentage points
    wins, ties, block_wins = 0, 0, 0
    for loss in ("mse", "l1", "charbonnier", "floc_1.2"):
        for budget in ("100", "200"):
            diff = mr[loss][budget]["mean"] - mb[loss][budget]["mean"]
            if diff > comparison_tolerance:
                wins += 1
            elif diff < -comparison_tolerance:
                block_wins += 1
            else:
                ties += 1
    assert (wins, ties, block_wins) == (7, 1, 0), (wins, ties, block_wins)
    assert abs(mr["floc_1.2"]["200"]["mean"] - mb["floc_1.2"]["200"]["mean"]) < comparison_tolerance
    print(f"  masking accuracy comparison (float-tolerant): random {wins} wins, {ties} tie, block {block_wins} wins")

    print("\nTokenizer: Conv1d(input=128,kernel=8,stride=4,padding=2), 32 tokens")
    print("  raw positions covered by one token: 4; by two tokens: 124")
    print(f"  exact-16 random-token masking: E[fully hidden] = {random_hidden:.4f}/128 = {random_hidden/128:.4%}")
    print(f"  Bernoulli(0.5) masking:         E[fully hidden] = {bernoulli_hidden:.4f}/128 = {bernoulli_hidden/128:.4%}")
    print(f"  contiguous-16 token block:     E[fully hidden] = {block_mean:.4f}/128 = {block_mean/128:.4%}")
    print("  contiguous-block starts: 15 hide 60 raw positions; 2 hide 62")
    print("\nPASS: frozen v70 results and tokenizer geometry match the audit.")

if __name__ == "__main__":
    main()
