# 26SP R1: Student-t stress-test fairness, exact five-pair statistics, and cross-distribution limits

Date: 2026-11-13. Status: prospective post-submission audit; frozen v70, submitted R1, and every archived accuracy remain unchanged.

## Evidence directly checked

- Source: `revision_v70_20261010/code/floc_msm_student_t.py`, frozen manifest SHA `c97a55dae882b43cbeec636f18172e4d54082fc340a24ec3f01830fc1f375cba`.
- Configuration: `revision_v70_20261010/results/student_t/config.json`, indicating `student_t_nu=1.5`, 5 downstream seeds, 10,000 unlabeled pretraining signals, three labeled budgets (50, 200, 500), 500 test examples, 20 pretraining epochs, `pretraining_noise_scale=0.15`, `downstream_noise_scale=0.30`.
- Full per-seed results: `revision_v70_20261010/results/student_t/results.json`.
- Comparator source: `revision_v70_20261010/code/floc_msm_main_corrected.py` and corrected 16-of-32 primary `results/main_table_reconciliation_20261009`.
- Manuscript unmarked R1 Section 4.5, assembled PDF page 85, states Student-t `nu=1.5` budget accuracy ranges and that Charbonnier leads at 50/200 while L1 leads at 500. These descriptions agree with the archive.

## Findings

### 1. Correct distribution implemented at every stage

The generator is `np.random.standard_t(args.nu,size=shape)*scale`, with `nu=1.5`, used at pretraining scale 0.15 and both labeled training/test scale 0.30, before per-sequence max-absolute scaling. There is no hidden α=1.5 CMS draw reused in the Student-t runner, and no distribution mismatch between pretraining and evaluation.

The Student-t distribution has a regularly varying tail with index ν=1.5, but is **not** an SαS law. Its scale multiplier 0.15/0.30 is not a guarantee of quantile-, median absolute deviation-, or tail-exceedance-matched corruption to α-stable noise with those nominal multipliers. After max-absolute scaling, the observed input sequences are bounded; injected Student-t tails do **not** prove an identical learned residual distribution.

### 2. A stronger within-Student-t loss comparison than across-distribution contrast

Within Student-t, the runner constructs the unlabeled corpus once, resets `torch.manual_seed(args.seed+1)` before each objective's SSL pretraining, creates a new identically-seeded loader (`args.seed+2`) for each loss, and resets `torch.manual_seed(seed); np.random.seed(seed)` before each downstream evaluation. This matches the same seed and synthetic downstream draw across losses at fixed label budget. Only **one SSL-pretrained encoder per loss** is trained; downstream SD is conditional on that encoder.

By contrast, the Student-t runner masks *each token independently with probability 0.5*, i.e. the number of masked tokens in a 32-token sequence has `Binomial(32,0.5)` distribution (mean 16, SD `sqrt(8)=2.828...`). The primary corrected α=1.5 comparison masks **exactly 16 of 32 tokens** per sequence, using uniform random ranking. This is a second difference beyond the noise family.

Also, both generators interleave clean waveform/jitter random draws with noise draws in the same NumPy stream; the Student-t sampling routine and CMS routine consume different random values. Even if the initial RNG seed is equal, **the unlabeled clean waveform mixtures and the Gaussian jitter realized after the first corruption are not kept fixed across the two runners**. The labeled classification templates remain fixed by class, but jitter and noise realizations are likewise not matched cross-distribution. Thus the Student-t experiment is **not** a controlled one-factor replacement of SαS by Student-t. Do not compare their accuracy gaps as causal differences attributable only to noise family.

Across labeled budgets, the generator creates a training set before test data using one reset NumPy stream. For each seed, the smaller labeled training sets are prefix-nested by construction, but budget-dependent RNG consumption changes the generated test sets. In addition, **500 labels receive 50 fine-tuning epochs, whereas 50 and 200 labels receive 30 epochs**. Thus the comparison is not a label-budget-only ablation with fixed test data and optimization duration.

### 3. Recomputed Student-t mean accuracies and p=1.2 vs MSE seed differences

| Labels | MSE | p=1.2 | p=1.2 minus MSE (pp) | p=1.2 positive paired seeds | Enumerated two-sided sign-flip p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 88.00% | 89.92% | +1.92 | 5/5 | 0.0625 |
| 200 | 92.16% | 95.24% | +3.08 | 5/5 | 0.0625 |
| 500 | 95.44% | 97.28% | +1.84 | 5/5 | 0.0625 |

In units of correct predictions on the same 500-test-example split per seed, the p=1.2 minus MSE paired differences are:
- 50 labels: `[11, 2, 12, 21, 2]` (sum 48 across five seeds).
- 200 labels: `[9, 26, 14, 16, 12]` (sum 77).
- 500 labels: `[10, 5, 13, 8, 10]` (sum 46).

These values exhaustively enumerate 32 possible sign assignments for each **five-pair budget**. The p-value `0.0625` is the smallest possible two-sided value when five nonzero differences all have the same sign. Formal exact null inference additionally relies on the appropriate symmetry/exchangeability condition for the paired differences. This test was not prospectively declared a primary Student-t hypothesis test. Do not claim statistical significance, and do not pool the 15 differences from five reused seed IDs across three correlated budgets as 15 independent trials.

### 4. The within-Student-t winner margins are also small

| Labels | Highest average accuracy | Next-best mean | Margin (pp) | Additional correct predictions across five 500-example tests |
| ---: | --- | --- | ---: | ---: |
| 50 | Charbonnier 90.16% | p=1.2 89.92% | 0.24 | 6/2500 |
| 200 | Charbonnier 95.48% | L1 and p=1.2 both 95.24% | 0.24 | 6/2500 |
| 500 | L1 97.44% | p=1.2 97.28% | 0.16 | 4/2500 |

At 200 labels, the L1 and p=1.2 means differ only in float32 roundoff (`~2.38e-8` fractional accuracy), with exactly the same integer number of correct predictions. Ranking should be based on meaningful accuracy resolution, not raw floating-point ordering.

As a result, the Student-t stress test provides directional evidence against **MSE** under this particular masked-learning pipeline, but no robust evidence that a single robust alternative (Charbonnier, L1 or p=1.2) is reliably best.

## Recommended manuscript/rebuttal update, if an editable R1 is requested

Suggested concise insert for Section 4.5:

> For the ν=1.5 Student-t sensitivity run, all losses share the same within-run pretraining corpus and matched downstream seeds, and the p=1.2 loss exceeds MSE in all five seeds at each tested budget (mean margins 1.84–3.08 percentage points). However, with only five paired downstream seeds, each of the three exhaustive two-sided sign-flip diagnostics has p=0.0625; these results remain descriptive. Moreover, the leading robust alternatives differ by only 0.16–0.24 points in mean accuracy. Unlike the exact-count masking in the primary corrected α-stable comparison, this extension uses Bernoulli token masks, and the two distribution-specific generators do not hold the clean waveforms/jitter realizations fixed. Hence between-distribution differences are not a one-factor noise-family ablation.

Optional short version:

> The Student-t experiment uses Bernoulli rather than fixed-count masking and generates separate noise-dependent synthetic streams; its results are therefore an independent stress test, not an isolated Student-t-versus-SαS distribution comparison.

Do not append this exploratory statistical diagnostic to already submitted revised PDFs. Avoid interpreting three small-sample p-values near 0.05 as a multiplicity-corrected finding. In future transparent reporting, the best approach is to retain the archived full per-seed results and disclose the independent noise/masking protocol rather than add a new suite of high-cost retraining experiments.

## Verification

Recomputed all 18 loss-by-budget result cells and Student-t pairwise enumerations from archived GitHub JSON in an independent code check; every source-level protocol claim was checked against the released runner. A standard-library-only checker is available at `revision_audits/audit_student_t_protocol_and_signflip.py`. The newly added Python checker has not yet been run from a full local repository clone. No experiment was rerun, no output JSON changed.
