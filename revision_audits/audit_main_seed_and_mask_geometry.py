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

def audit_mask_ratio_endpoints():
    """Data-only 25/75% Bernoulli ratio audit, including scratch negative control.

    The two archives have the same recorded runner digest and master seed,
    but this is NOT an independently verified bitwise matched comparison.
    Never attribute the observed difference uniquely to mask ratio.
    """
    base = ROOT / "revision_v70_20261010"
    script = (base / "code/provenance/r4_mask_ratio_audit.py").read_text(encoding="utf-8")
    assert "torch.rand(xb.shape[0],tok.shape[1],1,device=device)<CONFIG['mask_ratio']" in script
    assert "if loss_type=='scratch': continue" in script
    assert "eval_downstream(None,n_labeled" in script
    assert script.count("CONFIG['mask_ratio']") == 1
    assert "torch.manual_seed(seed); np.random.seed(seed)" in script
    configs = {}
    result = {}
    for folder, q in (("025", .25), ("075", .75)):
        folder_path = base / "results/mask_ratio" / folder
        cfg = json.loads((folder_path / "config.json").read_text(encoding="utf-8"))
        configs[folder] = cfg
        result[folder] = json.loads((folder_path / "results.json").read_text(encoding="utf-8"))
        assert cfg["mask_ratio"] == q and cfg["n_seeds"] == 5
        assert cfg["n_labeled"] == [50,200,500] and cfg["n_test"] == 500
        assert cfg["losses"] == ["scratch","mse","l1","huber","floc_1.2"]
        # Four positions have 1-token coverage, 124 have 2-token coverage.
        # Independent token masking gives E(hidden)=4*q + 124*q*q.
        concealed = 4*q + 124*q*q
        assert abs(concealed - {".25":8.75,".75":72.75}[str(q)]) < 1e-12
        print(f"Bernoulli q={q:.2f}: mean masked tokens={32*q:g}/32, "
              f"E[directly fully concealed raw coordinates]={concealed:g}/128 "
              f"({concealed/128:.4%})")
    assert configs["025"]["code_sha256"] == configs["075"]["code_sha256"]
    for key in ("seed","alpha","pt_epochs","pt_samples",
                "n_test","ft_epochs","scratch_epochs"):
        assert configs["025"][key] == configs["075"][key]

    # All differences are integer-accurate reaggregations of 5*500
    # test predictions (one independent downstream evaluation per seed).
    def correct_counts(folder, loss, budget):
        vals = result[folder][loss][str(budget)]["seeds"]
        assert len(vals) == 5
        out = [round(v*500) for v in vals]
        assert all(abs(v-k/500)<1e-7 for v,k in zip(vals,out))
        return out

    expected_gap = {
        "mse": (4.12, .96, .24),
        "l1": (6.0, 2.04, .36),
        "huber": (3.68, 1.16, .44),
        "floc_1.2": (4.76, 2.32, .40),
        "scratch": (1.24, .92, 0.0),
    }
    changed_scratch_seeds = 0
    for loss, gaps in expected_gap.items():
        for budget,gap in zip((50,200,500),gaps):
            counts_a=correct_counts("025",loss,budget)
            counts_b=correct_counts("075",loss,budget)
            actual=(sum(counts_a)-sum(counts_b))/25
            assert abs(actual-gap) < 1e-12, (loss,budget,actual,gap)
            print(f"{loss:>9} at {budget:>3} labels: 25%-minus-75% "
                  f"test mean={actual:+.2f} pp")
            if loss=="scratch":
                changed_scratch_seeds += sum(a!=b for a,b in zip(counts_a,counts_b))
    assert changed_scratch_seeds == 14
    print("PASS: scratch (which does not use a mask or pretrained weights)")
    print("      differs at 14/15 matched-index seed/budget records between")
    print("      the separate 25% and 75% archived runs; even at 500 labels")
    print("      where mean accuracies tie, four of five seed results differ.")
    for ratio in ("025","075"):
        p_200=sum(correct_counts(ratio,"floc_1.2",200))
        mse_200=sum(correct_counts(ratio,"mse",200))
        assert abs((p_200-mse_200)/25 - (0.28 if ratio=="025" else -1.08)) < 1e-12
    print("PASS: at 200 labels p=1.2-minus-MSE is +0.28 pp at 25%,")
    print("      but -1.08 pp at 75%. A runner-global execution disparity")
    print("      also exists, so this is NOT an isolated mask-ratio effect.")
    print("No GPU run, retrospective best-parameter selection or archive edit.")


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
    audit_mask_ratio_endpoints()

    print("\nPASS: frozen v70 results and tokenizer geometry match the audit.")

if __name__ == "__main__":
    main()
