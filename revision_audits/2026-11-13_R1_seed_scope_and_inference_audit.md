# SIGPRO-D-26-02622 R1: main-seed scope, test-set pairing, and restricted multiplicity correction

Date: 2026-11-13.
Status: post-R1 prospective quality audit. The assembled R1 manuscript and frozen `revision_v70_20261010` archive **remain unchanged**.
Primary files:
- `revision_v70_20261010/code/floc_msm_main_corrected.py`
- `revision_v70_20261010/code/provenance/run_main_table_reconciliation_executed_20261009.py`
- `revision_v70_20261010/results/main_table_reconciliation_20261009/results.json`
- `revision_v70_20261010/code/alpha1_training_dynamics/targeted_alpha10_fullseed.py`
- `revision_v70_20261010/code/audit_cauchy_unpaired_welch.py`
- R1 Editorial Manager assembly, **unmarked manuscript**, PDF pages 76–78 and 85 (not the deleted material displayed in marked-up pages).

## Finding A. The main five seeds vary synthetic test data, not only downstream initialization

In `floc_msm_main_corrected.py`, per budget and per downstream seed, the runner resets

    torch.manual_seed(seed); np.random.seed(seed)

and calls

    eval_downstream(weights, n_labeled, CONFIG['n_test'], ...)

which constructs a new encoder/head and then calls

    Xtr,ytr = gen_cls(n_train,128,1.5)
    Xte,yte = gen_cls(n_test,128,1.5)

The `gen_cls` function draws a separate Gaussian jitter vector and corrected CMS noise for each generated sequence. The pretraining corpus is generated once before the loss loop, and **one pretrained checkpoint per loss** is retained across all budgets and five downstream seeds.

Consequences:

1. **Within a given budget and seed**, all losses restart NumPy and Torch random streams consistently, so their synthetic train/test realizations match. The paired five-seed sign-flip contrast remains substantively appropriate for those matched downstream experiments.
2. **Across downstream seeds** within the same budget, the training sequences, 500-sample test sequences and noise draws, classification head initialization, and training/shuffling randomness change. Thus the reported SD is conditional on the pretrained checkpoint but **does not isolate fine-tuning randomness**.
3. **Across label budgets**, the generator *regenerates* training data after the same NumPy-seed reset, but its sequential per-example draws and rowwise normalization mean that the smaller training sample set is **prefix-nested within the larger training sample set for a fixed seed**. However, the 500-example test set **is not fixed**: `gen_cls(n_train)` consumes a budget-dependent number of NumPy variates before `gen_cls(n_test)` runs. Moreover, the 500-label budget uses **50 fine-tuning epochs**, whereas the 50/100/200-label budgets use **30**, so this is not a training-size-only ablation with a common test realization or fixed training duration.
4. This does not make the main within-budget loss comparison invalid. It is a precision/reproducibility qualification; the current unmarked manuscript's "five matched fine-tuning seeds" is incomplete about what changed.

### Proposed exact replacement (Section 4.1, unmarked R1 PDF page 76)

Current wording (paraphrased): the corrected α=1.5 benchmark uses one checkpoint per loss and five matched fine-tuning seeds, with SD reflecting downstream variability conditional on that checkpoint.

Proposed future manuscript sentence:
> For the corrected α=1.5 core benchmark, each loss is pretrained once. Within each labeled budget, five matched downstream seeds regenerate synthetic training and 500-example test sets (including fresh impulsive noise) and reinitialize and fine-tune the classifier, while retaining the loss-specific pretrained encoder. The same generated train/test pair is used for every loss at a given budget and seed, supporting paired loss comparisons. Test draws are not shared across different label budgets; the reported standard deviations therefore describe conditional downstream data-generation and training variability, rather than pretraining variability or optimization noise alone.

### Proposed exact replacement (Table 1 caption, unmarked R1 PDF page 78)

> Corrected-CMS α=1.5 comparison: test accuracy (percent), mean ± population standard deviation over five matched downstream data-generation and training seeds, conditional on one pretrained checkpoint per loss. The synthetic train/test realizations match across losses within a label budget and seed, but test samples are regenerated across budgets. The p=1 row is omitted because its reconstruction objective equals L1.

The masking-sensitivity subsection also describes "fine-tuning, not pretraining, variability" (unmarked R1 page 84), but the same main-runner `eval_downstream` procedure implies variation in synthetic examples and their test noise, **not only the fine-tuning process**, if that runner generated its results. The sensitivity provenance should be checked against its exact runner before changing that subsection.

## Finding B. Exact five-seed sign-flip p-values can be reported instead of only the resolution bound

From the frozen main reconciliation `results.json`, enumerating all 32 sign assignments to paired per-seed differences:

| Labels | p=1.2 minus MSE mean (pp) | 2-sided exact sign-flip p vs MSE | 2-sided exact sign-flip p vs L1 |
| --- | ---: | ---: | ---: |
| 50 | +0.44 | 0.5625 | 0.6250 |
| 100 | +1.52 | 0.1875 | 0.1875 |
| 200 | +1.24 | 0.1875 | 0.8125 |
| 500 | +1.44 | 0.1875 | 0.8750 |

The p=1.2 versus L1 comparisons use the same five matched seed IDs and are also descriptive. The 500-label difference against L1 contains one zero difference; the enumeration keeps that sign slot, yielding 0.8750. No hypothesis is significant at conventional 0.05 per-comparison level. With only five pairs the lowest attainable nonzero two-sided permutation value under this test is 0.0625.

**Recommended prose**: retain the concise "minimum possible p=0.0625" statement in the abstract but, if space permits, put the four actual p values in the main comparison paragraph or a Supplementary note. This avoids a reader mistaking the resolution limitation for an absence of computation.

New reproducible (standard-library) verifier:
`revision_audits/audit_main_seed_scope_and_signflip.py`

## Finding C. Targeted α=1 Welch + Holm findings are conditional on a limited family

Archived script `audit_cauchy_unpaired_welch.py` defines four independent-sample, two-sided Welch comparisons of **fixed p=1.2 versus MSE**, at budgets 50/100/200/500, and applies Holm step-down correction across precisely **those four hypotheses**.

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

## Finding E. Two same-source primary archives differ numerically; R1 Table 1 has a uniquely identified source

**Additional provenance audit (2026-11-13).** The frozen release contains **two distinct** α=1.5 main-series result archives:

1. `results/main_alpha15/` — the earlier, broader run containing nine loss identifiers (including scratch and learnable p).
2. `results/main_table_reconciliation_20261009/` — the four-loss core run with `mse,l1,huber,floc_1.2`, expressly **superseding the former for manuscript Table 1**. This authority is stated both in the frozen README and in the reconciliation `run_metadata.json` `notes` field. The unmarked submitted R1 Table 1 (assembled PDF page 78) matches **this second run**, to its printed one-decimal accuracy/SD precision.

Both configuration files identify the **same main source SHA-256** `880fbd11139d81d5abe462cd5190a28c973dee1c54e7932d7f629a9c7b541104`, master seed `20261007`, α=1.5, 20 pretraining epochs, five downstream seeds, 500 test examples, and exact-count 16-of-32 random-token masks. Both run metadata files mark `deterministic_algorithms=true` and record the same PyTorch `2.11.0+cu128`, NumPy `2.5.2` and GPU `RTX 5060 Ti`. The reconciliation wrapper additionally records CuDNN deterministic=true and benchmark=false; the earlier run metadata does not record those backend flags. Importantly, loss-list lengths/order differ (nine versus four). **The fact that two metadata records call a run deterministic is not proof that they produced bitwise-identical numerical trajectories under fully identical execution conditions.**

Despite common source hash and seed, all **16** overlapping loss–budget means differ:

| Loss | 50 labels: reconciliation minus earlier (pp) | 100 (pp) | 200 (pp) | 500 (pp) |
| --- | ---: | ---: | ---: | ---: |
| MSE | −0.64 | −0.44 | −0.12 | −0.28 |
| L1 | +0.64 | +0.52 | −0.08 | +0.32 |
| Huber δ=1 | **−1.24** | −0.92 | −0.08 | −0.64 |
| p=1.2 | −0.32 | +0.12 | +0.28 | +0.36 |

All 16 entries were reconciled from **integer correct-prediction counts** in 5×500 test evaluations, not potentially deceptive rounded mean values. Even pretraining epoch-20 mean objectives differ: MSE **0.03868864** (earlier) versus **0.03885275** (reconciliation), and p=1.2 **0.09849574** versus **0.09851298**. The runs are **not numerically identical GPU executions**, despite the same recorded base source digest.

The substantive p=1.2–MSE mean gain is positive at all four budgets in **both** archives:

| Labels | Earlier gain (pp) | R1 Table 1 reconciliation gain (pp) |
| ---: | ---: | ---: |
| 50 | +0.12 | **+0.44** |
| 100 | +0.96 | **+1.52** |
| 200 | +0.84 | **+1.24** |
| 500 | +0.80 | **+1.44** |

The top robust-loss ordering is not identical in every cell. In particular p=1.2 versus L1 at 200 labels is **−0.20 pp** in the earlier run but **+0.16 pp** in the reconciliation. This demonstrates **run-conditional ranking fragility**, not statistical significance or a reproducibility failure attributable to a specific cause. There are only two archived runs with different loss lists; they do **not** estimate an independent-pretraining distribution or isolate CuDNN, GPU determinism, order effects or software provenance as the reason for the differences.

**Conclusion for the paper:** R1's Table 1 and its +0.4–1.5 pp effect-range wording reproduce the designated reconciliation archive. Therefore **no internal manuscript–Table 1 numeric contradiction** was found. For any quantitative reproduction, the preferred provenance target is **`main_table_reconciliation_20261009`**, not a silently merged selection of whichever numbers appear better in the earlier `main_alpha15`.

### Current-path defect in the historical executed wrapper

At the release's present tree location, `code/provenance/run_main_table_reconciliation_executed_20261009.py` contains:

```python
root = Path(__file__).resolve().parents[1]
runner = Path(__file__).with_name("floc_msm_main_corrected.py")
```

From its **currently archived** `code/provenance` directory, this constructs the nonexistent `code/provenance/floc_msm_main_corrected.py` and an output root under `code/results/` rather than `revision_v70_20261010/results/`. The archived copy therefore **cannot be rerun in place**. This is a path/packaging defect after archival relocation, **not** evidence that the historically executed wrapper failed when run at its then-current location. It does not change any saved numerical outputs or falsify the run metadata.

The **actual usable source entrypoint** is `code/run_main_table_reconciliation.py`, located adjacent to `code/floc_msm_main_corrected.py`; its sibling path resolves correctly. It supports `--output-dir`. To keep frozen v70 unchanged in future reproduction work, **explicitly choose a new absolute output directory outside the frozen tree**; note that the entrypoint's no-argument default creates a new `results/reproduction` subdirectory *inside* the frozen release. Do **not** edit either frozen wrapper to conceal this packaging history.

The existing standard-library `revision_audits/audit_main_seed_scope_and_signflip.py` now also checks all 16 archived numeric differences, source hashes, explicit supersession and existence of the usable vs nonexistent current runner paths, without importing Torch or rerunning any model. Its new source checks have been independently verified against live GitHub files, but the augmented Python script has **not yet run from a full local checkout**.

**Disposition:** one definite archived-wrapper path defect; one reproducibility/variance caveat; **no** evidence of numerical misreporting in R1 Table 1 and **no** justification for replacing published values or starting a new GPU sweep just to re-rank ties.

## Finding F. Stronger control: cross-run differences also occur without any self-supervised pretraining

The cross-run main-table discrepancy above **cannot be explained solely by loss-list order or pretrained-encoder variation**. The main source resets PyTorch and NumPy seeds before every downstream call, and the scratch branch invokes `eval_downstream(None, ...)`, which initializes an encoder from scratch without loading pretrained weights. At each label budget and seed, the same code regenerates the synthetic training/test arrays. Their original array bytes were **not archived**, so exact cross-run input equality has not been checked independently, though the source and NumPy version are recorded as the same.

Recomputing the five-seed correct counts in the **scratch** arm yields:

| Total labels | Earlier scratch mean ± population SD (%) | R1 Table 1 reconciliation scratch mean ± SD (%) | Reconciliation minus earlier (pp) |
| ---: | ---: | ---: | ---: |
| 50 | 81.32 ± 1.27 | 81.56 ± 4.37 | **+0.24** |
| 100 | 76.44 ± 21.06 | 80.16 ± 15.91 | **+3.72** |
| 200 | 90.60 ± 1.08 | 90.16 ± 1.81 | **−0.44** |
| 500 | 96.56 ± 1.28 | 96.84 ± 1.26 | **+0.28** |

At **100 total labels, seed index 2**, scratch correct predictions change from **175/500 (35.0%)** to **242/500 (48.4%)**, a **13.4-percentage-point** difference at that one test realization. This single seed accounts for a substantial portion, but not all, of the +3.72-point shift in the across-seed mean. In the archived source, scratch uses 80 supervised epochs and is evaluated with the same code path in both archives. Its instability makes the *numerical size* of a 'pretraining versus scratch' gap especially run-conditional. It does not invalidate using the designated R1 values as a descriptive historical result.

A second source-level control is the **first trained loss**: the earlier config begins with `scratch,mse,...`, but the runner **skips** `scratch` in its pretraining loop. The reconciliation config begins with `mse,...`. Therefore `mse` is the **first actual SSL pretraining** in both. The two archived MSE first-epoch mean losses are already nonidentical: `0.0684799393521` versus `0.0684800391333` (absolute difference ~9.98×10⁻⁸). This shows the initial divergence occurs **before any earlier trained loss can affect global stochastic state**. It does not alone identify whether different kernel behavior, environment, unrecorded flags, hardware runtime, artifact provenance or other causes are responsible.

Both runs record `torch.use_deterministic_algorithms` status as **true** and list the same Python/PyTorch/NumPy/GPU family. Only the reconciliation `run_metadata.json` explicitly records `cudnn_deterministic=true` and `cudnn_benchmark=false`. The older metadata lacks these fields; **absence of a recorded value must not be interpreted as proof that a flag was false**. Source SHA and deterministic-algorithm status are not complete byte-for-byte execution fingerprints.

**What can be concluded:** at least one uncontrolled or unobserved execution/provenance dimension separates the two saved numerical runs, or the artifacts were not created under otherwise identical conditions. Existing archive outputs cannot identify which. Neither the different loss list nor prior encoder pretraining alone explains the scratch discrepancy. It would be scientifically unsound to report this as proven GPU nondeterminism or as misconduct, and wrong to merge/average/choose the better table.

**Minimal future reproduction log, if an authorized independent run is ever required:** use the canonical runner in an output directory outside frozen v70, and save hashes of (i) unlabeled pretraining inputs, (ii) at least one downstream 500-example test tensor for each budget/seed, (iii) freshly initialized model parameters, (iv) epoch-1 / epoch-20 checkpoint state dicts; also record complete CUDA, cuDNN, driver and TF32/determinism flags, environment variables related to CUDA determinism, and the exact command line. These discriminators are needed to distinguish data, initialization, numeric runtime and optimizer trajectories. This is a reproducibility *recommendation*, not a request to execute a new GPU sweep now.

The `revision_audits/audit_main_seed_scope_and_signflip.py` checker was expanded in place to assert the scratch shifts, the 100-label seed-2 count change, the first-trained-MSE property and the difference in metadata completeness. Live archived JSON and source text were independently rechecked through the repository connector; the updated Python script has **not been run in a full local checkout**.

**Severity:** reproducibility/provenance investigation warranted if an editor or external reproducer requests precise rerun matching; **no demonstrated Table 1 transcription error, no established statistical significance improvement, no reason to alter submitted R1.**

## Audit disposition and prioritization

- Main numerical accuracy figures remain unchanged; within-budget main comparisons are matched and valid as descriptive evidence.
- **Highest priority manuscript correction**: expand the scope of "fine-tuning-seed variability" to include newly drawn training/testing synthetic data, and disclose that test sets are not held constant across budgets.
- Medium priority: include actual main exact test p values in supplemental or Section 4.2.
- Medium priority: clarify that α=1 Holm tests correct only four p=1.2–MSE contrasts.
- Low priority: revisit sensitivity text once its provenance is checked.
- Neither this note nor its companion verifier re-trains a model, and neither changes the R1 PDF or the frozen v70 results.


## Finding D. Masking-sensitivity replication scope is now source-confirmed; missing random-arm source is already disclosed

We further inspected `code/provenance/mask_strategy_contiguous_block_v2_runner.py`,
`results/masking/random_token_exact/config.json`,
`results/masking/contiguous_block_v2/config.json`,
and the **frozen v70 README**.

- The contiguous-block runner uses the *same* `eval_downstream` implementation as the canonical main runner: for each budget and seed it regenerates `gen_cls(n_train)` followed by `gen_cls(n_test)`, after resetting NumPy and Torch seeds. Thus the R1 masking paragraph's wording “the SD captures fine-tuning, not pretraining, variability” is **incomplete**: it includes the newly generated downstream training/test data and noise.
- Random-token config records `code_sha256=5b0315ff29b8eaff269f3902c24a94e55741bb265ead2a6a44705b43aea2f803`; contiguous-block config records `880fbd11139d81d5abe462cd5190a28c973dee1c54e7932d7f629a9c7b541104`. These are **different executed script digests**.
- The frozen `revision_v70_20261010/README.md` explicitly says the historical exact-count random-token masking runner **was not recovered**; its archived configuration/output are not a fully rerunnable source release. Therefore the provenance gap was already disclosed, not newly discovered in this audit.
- Archive results remain suitable for describing which group has a higher mean. Without source equality/diff validation of the missing random-token runner, the observed difference must **not** be asserted as a clean causal mask-topology-only effect.

Proposed precision sentence for next editable Section 4.4:
> In the archived exact-count masking comparison, the random-token runs had higher mean accuracy in seven of eight loss–budget cells, with one tie. These are descriptive between-run differences conditional on the recorded protocols; the random-token arm's original executable source was not recovered in the public archive, so exact source equivalence cannot be independently verified. Each arm's downstream seeds also regenerate synthetic training/test examples, making its reported SD conditional on the pretrained encoder but not restricted to fine-tuning optimization variability.

This addition **does not** allege that the different SHA values correspond to a material algorithm change. The difference merely prevents verification of strict source-equivalence without the missing runner.

## Maintenance correction: training prefix nesting and epoch budget

An earlier version of item 3 incorrectly asserted that the independently invoked smaller-budget training datasets were not nested. In fact, because `gen_cls` draws each example sequentially, the same seed produces identical **training prefixes** across label budgets. The associated test sets are still budget-dependent, and the 500-label case uses 50 rather than 30 fine-tuning epochs. The latter two factors prevent interpreting cross-budget accuracy differences as an isolated sample-size response. This correction affects prose only; the recorded results and within-budget paired comparisons are unchanged.

## Maintenance correction: finite-enumeration p values and their assumptions

The five-pair sign-flip procedure enumerates all 2^5 sign configurations, and the p values above are **computationally exact for this statistic**. Formal exact Type-I error control is conditional on sign-exchangeability/symmetry of the paired differences under the relevant null (for example, the sharp null of no loss effect under a genuinely randomizable treatment assignment). Matching pseudorandom seeds and using common synthetic data across losses does not, by itself, establish this stochastic condition. Therefore treat the resulting p values as transparent finite-sample descriptive diagnostics, **not** distribution-free confirmatory evidence under arbitrary composite nulls. The observation that no possible p value is below 0.0625 for five nonzero paired differences still holds as a combinatorial resolution limit.

The reconstructed two-sided p values should not be described as a formal equivalence test, nor should the absence of p < 0.05 be taken as evidence that losses are equivalent. In any prospective revision, use wording such as "exhaustive sign-flip diagnostic" when the exchangeability assumption has not been established.

This note supersedes any more categorical characterization of the five-seed sign-flip calculations elsewhere in the audit set; it changes neither the data nor the computed p values.

## Maintenance verification: a floating-point pseudo-win in masking comparisons

An independent reaggregation of the archived masking arrays initially found eight strictly positive random-minus-block mean differences when using **bitwise floating-point `>`**. The eighth difference (p=1.2, 200 labels) is only `0.00000119209` **percentage points** (mean accuracy difference `1.19209e-8`) and arises from finite-precision float32 accuracy aggregation. It is far below the accuracy reporting resolution. With an explicit tolerance of `1e-7` in fractional accuracy (0.00001 percentage points), the correct descriptive summary remains **seven higher random-token means, one numerical tie, zero block wins**. No archived accuracies or paper percentages change.

The companion `audit_main_seed_and_mask_geometry.py` script now asserts the 7/1/0 grouping using that tolerance. Rigid float equality should not be used for these summaries.
