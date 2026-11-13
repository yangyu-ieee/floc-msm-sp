# SIGPRO-D-26-02622 R1: main-seed scope, test-set pairing, and restricted multiplicity correction

Date: 2026-11-13.
Status: post-R1 prospective quality audit. The assembled R1 manuscript and frozen \`revision_v70_20261010\` archive **remain unchanged**.
Primary files:
- \`revision_v70_20261010/code/floc_msm_main_corrected.py\`
- \`revision_v70_20261010/code/provenance/run_main_table_reconciliation_executed_20261009.py\`
- \`revision_v70_20261010/results/main_table_reconciliation_20261009/results.json\`
- \`revision_v70_20261010/code/alpha1_training_dynamics/targeted_alpha10_fullseed.py\`
- \`revision_v70_20261010/code/audit_cauchy_unpaired_welch.py\`
- R1 Editorial Manager assembly, **unmarked manuscript**, PDF pages 76–78 and 85 (not the deleted material displayed in marked-up pages).

## Finding A. The main five seeds vary synthetic test data, not only downstream initialization

In \`floc_msm_main_corrected.py\`, per budget and per downstream seed, the runner resets

    torch.manual_seed(seed); np.random.seed(seed)

and calls

    eval_downstream(weights, n_labeled, CONFIG['n_test'], ...)

which constructs a new encoder/head and then calls

    Xtr,ytr = gen_cls(n_train,128,1.5)
    Xte,yte = gen_cls(n_test,128,1.5)

The \`gen_cls\` function draws a separate Gaussian jitter vector and corrected CMS noise for each generated sequence. The pretraining corpus is generated once before the loss loop, and **one pretrained checkpoint per loss** is retained across all budgets and five downstream seeds.

Consequences:

1. **Within a given budget and seed**, all losses restart NumPy and Torch random streams consistently, so their synthetic train/test realizations match. The paired five-seed sign-flip contrast remains substantively appropriate for those matched downstream experiments.
2. **Across downstream seeds** within the same budget, the training sequences, 500-sample test sequences and noise draws, classification head initialization, and training/shuffling randomness change. Thus the reported SD is conditional on the pretrained checkpoint but **does not isolate fine-tuning randomness**.
3. **Across label budgets**, the same seed does not guarantee the same test set: \`gen_cls(n_train)\` consumes a budget-dependent number of NumPy variates before \`gen_cls(n_test)\` runs. The training datasets are also newly generated rather than a nested finite sample. Do not interpret accuracy changes across budgets as a controlled nested-training-size ablation with common test realizations.
4. This does not make the main within-budget loss comparison invalid. It is a precision/reproducibility qualification; the current unmarked manuscript's "five matched fine-tuning seeds" is incomplete about what changed.

### Proposed exact replacement (Section 4.1, unmarked R1 PDF page 76)

Current wording (paraphrased): the corrected α=1.5 benchmark uses one checkpoint per loss and five matched fine-tuning seeds, with SD reflecting downstream variability conditional on that checkpoint.

Proposed future manuscript sentence:
> For the corrected α=1.5 core benchmark, each loss is pretrained once. Within each labeled budget, five matched downstream seeds regenerate synthetic training and 500-example test sets (including fresh impulsive noise) and reinitialize and fine-tune the classifier, while retaining the loss-specific pretrained encoder. The same generated train/test pair is used for every loss at a given budget and seed, supporting paired loss comparisons. Test draws are not shared across different label budgets; the reported standard deviations therefore describe conditional downstream data-generation and training variability, rather than pretraining variability or optimization noise alone.

### Proposed exact replacement (Table 1 caption, unmarked R1 PDF page 78)

> Corrected-CMS α=1.5 comparison: test accuracy (percent), mean ± population standard deviation over five matched downstream data-generation and training seeds, conditional on one pretrained checkpoint per loss. The synthetic train/test realizations match across losses within a label budget and seed, but test samples are regenerated across budgets. The p=1 row is omitted because its reconstruction objective equals L1.

The masking-sensitivity subsection also describes "fine-tuning, not pretraining, variability" (unmarked R1 page 84), but the same main-runner \`eval_downstream\` procedure implies variation in synthetic examples and their test noise, **not only the fine-tuning process**, if that runner generated its results. The sensitivity provenance should be checked against its exact runner before changing that subsection.

## Finding B. Exact five-seed sign-flip p-values can be reported instead of only the resolution bound

From the frozen main reconciliation \`results.json\`, enumerating all 32 sign assignments to paired per-seed differences:

| Labels | p=1.2 minus MSE mean (pp) | 2-sided exact sign-flip p vs MSE | 2-sided exact sign-flip p vs L1 |
| --- | ---: | ---: | ---: |
| 50 | +0.44 | 0.5625 | 0.6250 |
| 100 | +1.52 | 0.1875 | 0.1875 |
| 200 | +1.24 | 0.1875 | 0.8125 |
| 500 | +1.44 | 0.1875 | 0.8750 |

The p=1.2 versus L1 comparisons use the same five matched seed IDs and are also descriptive. The 500-label difference against L1 contains one zero difference; the enumeration keeps that sign slot, yielding 0.8750. No hypothesis is significant at conventional 0.05 per-comparison level. With only five pairs the lowest attainable nonzero two-sided permutation value under this test is 0.0625.

**Recommended prose**: retain the concise "minimum possible p=0.0625" statement in the abstract but, if space permits, put the four actual p values in the main comparison paragraph or a Supplementary note. This avoids a reader mistaking the resolution limitation for an absence of computation.

New reproducible (standard-library) verifier:
\`revision_audits/audit_main_seed_scope_and_signflip.py\`

## Finding C. Targeted α=1 Welch + Holm findings are conditional on a limited family

Archived script \`audit_cauchy_unpaired_welch.py\` defines four independent-sample, two-sided Welch comparisons of **fixed p=1.2 versus MSE**, at budgets 50/100/200/500, and applies Holm step-down correction across precisely **those four hypotheses**.

The archived script's expected p-values, from its directly coded assertions:

| Labels | Welch raw p | Holm p across four p=1.2 vs MSE budgets |
| --- | ---: | ---: |
| 50 | 1.09e-5 | 2.18e-5 |
| 100 | 4.41e-7 | 1.32e-6 |
| 200 | 2.62e-8 | 1.05e-7 |
| 500 | 2.47e-5 | 2.47e-5 |

These values support the unmarked R1 statement "Holm-adjusted p < 0.001" **for this contrast family**. They do **not** adjust for all p choices, all strong robust baselines (L1/Charbonnier), all datasets, or the preceding exploratory choice of p=1.2. Nor does the ten-run targeted experiment prove a uniquely optimal fractional loss. The targeted alpha=1 runner uses loss-specific independent random streams, so the Welch analysis is consistent with its unpaired implementation; do not silently substitute a paired test.

Suggested addendum for Section 4.5 (unmarked R1 PDF page 85):
> The Holm adjustment covers the four prespecified-for-this-audit p=1.2-versus-MSE label-budget comparisons, not the broader exploratory selection among losses and protocols. These targeted tests therefore support a loss-specific contrast with MSE under the stated synthetic Cauchy-level protocol, rather than confirmatory superiority over the robust-loss family.

Avoid the word "preregistered" or a global familywise-error-control claim unless contemporaneous preregistration evidence exists. The program explicitly tracks these as *targeted* comparisons.

## Audit disposition and prioritization

- Main numerical accuracy figures remain unchanged; within-budget main comparisons are matched and valid as descriptive evidence.
- **Highest priority manuscript correction**: expand the scope of "fine-tuning-seed variability" to include newly drawn training/testing synthetic data, and disclose that test sets are not held constant across budgets.
- Medium priority: include actual main exact test p values in supplemental or Section 4.2.
- Medium priority: clarify that α=1 Holm tests correct only four p=1.2–MSE contrasts.
- Low priority: revisit sensitivity text once its provenance is checked.
- Neither this note nor its companion verifier re-trains a model, and neither changes the R1 PDF or the frozen v70 results.
