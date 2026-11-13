# 26SP R1: external evidence is not uniform — CWRU random-stream control and NOISEX realism

Audit date: 2026-11-13. Status: read-only evidence audit, submitted R1 and frozen `revision_v70_20261010` unchanged.

## High-priority conclusion

The unmarked R1 manuscript **already accurately reports** that the CWRU load-transfer results are near four-class chance (25%; best five-seed means 27.4%–31.1%), that NOISEX loss winners are mixed, and that the work does not establish a universally preferred loss. **No manuscript numerical contradiction was found.**

The new and important caveat is internal control: unlike the corrected core and NOISEX full-pipeline comparisons, the CWRU script does **not** reseed Torch RNG or create independently seeded DataLoaders between *loss objectives*. It produces different encoder initializations, pretraining masks, minibatch orders, classifier-head initializations and fine-tuning shuffle streams for different methods, even when using the same outer pipeline seed. The CWRU objective ranking is therefore a **confounded descriptive ranking**, not a common-random-number contrast isolating the loss.

## 1. CWRU source split and absolute classification performance

Checked `revision_v70_20261010/code/cwru_loss_compare_sourceclosed.py` and `results/cwru/{metadata.json,results.json}`.

- The archived configuration uses four classes (`Normal, IR007, B007, OR007@6`), drive-end (DE) vibration, train motor loads 0/1/2 hp vs held-out load 3 hp, `alpha=1.0` corrected symmetric Cauchy perturbations at nominal scale 0.30, and 10/20/50 **per-class** labeled training examples.
- The metadata lists **16 distinct NPZ inputs**, 12 train-load files (RPM 1797/1772/1750) and four test-load files (RPM 1730). The files are disjoint by name/load. Their SHA-256 metadata is archived, though we did **not** independently download the proprietary/input NPZ files or reconfirm their bytes here.
- For each source file, 1,024-point time segments are sampled at stride 512, giving **50% nominal within-file overlap**. This overlap is within the train group or within the test group, *not itself evidence of the same original file appearing in train and test*. It means the 953 held-out test windows are dependent and must not be treated as 953 independent source records.
- Train: **2,850** segments. Test: **953** segments. Test-class counts computed from every archived confusion matrix are **240,239,236,238**, totaling 953. Four-way uniform chance is 25%; the largest-class always-predict baseline is `240/953=25.18363%`.
- Perturbations are generated anew for each outer seed and inputs are z-scored **per noisy sequence** afterward. The same corrupted train/test tensors and per-class few-shot indices are available to all losses **within each outer seed**; that does not guarantee comparable Torch-model randomness.

Archived best-condition means:

| Per-class labels | Highest five-seed mean | Accuracy | Excess over test largest-class baseline |
| ---: | --- | ---: | ---: |
| 10 | L1 | **27.408%** | +2.225 pp |
| 20 | Charbonnier | **28.541%** | +3.358 pp |
| 50 | p=1.2 | **31.102%** | +5.918 pp |

At 50 labels/class, p=1.2 exceeds MSE (24.302%) by **6.800 pp** in five seedwise runs, but the resulting **31.102% absolute accuracy remains weak**. The positive difference and the low absolute level must both be reported. This condition cannot validate an industrial-grade fault diagnostic system.

### RNG fairness source evidence

The source calls `set_seed(seed)` only once at the start of each outer seed, before making the noisy datasets. It then executes:

```python
for lt in losses:
    if lt == "scratch": continue
    weights[lt] = pretrain_msm(x_pt, lt, args, device)
# ...
for shots in shots_list:
    for lt in losses:
        acc, conf = eval_classifier(..., seed)
```

Neither `pretrain_msm()` nor `eval_classifier()` resets Torch seeds. Both instantiate new random encoder/decoder/head/mask parameters and shuffled DataLoaders (without independent, fixed loader generators). The `eval_classifier()` function **does** use `np.random.default_rng(seed+10007*shots)` to select equal few-shot indices across losses, so that dimension is controlled.

This is **not data leakage**, and all reported means remain literal results of the archived code. It is a **loss-comparison design limitation**, particularly relevant given how close the absolute accuracies are to chance. This is different from the NOISEX design, which deliberately calls `seed_all(derive_seed(...,"pretrain-model"))` before every loss, and `seed_all(fit_seed)` before each fine-tuning.

The script uses 12 SSL epochs and 25 fine-tuning epochs, while scratch gets 35 epochs; differences in training budget and optimizer gradient clipping between external and primary experiments further forbid quantitative pooling.

## 2. NOISEX-92/UCR full-pipeline repetitions and dataset-wise heterogeneity

Checked `revision_v70_20261010/code/noisex92_ucr_seeded.py` and `results/noisex/{results.json,summary.json,run_metadata.json}`. The run records original UCR 2018 TRAIN/TEST TSV SHA-256, NOISEX WAV SHA-256, source code SHA-256, environment and GPU. **Raw WAV/TSV content was not downloaded or rehashed during this audit.**

- Eight UCR datasets, two per-class budgets (`5,10`) and three SSL loss choices (MSE, L1, p=1.2) plus scratch produce 16 dataset/budget comparisons, each with five **full-pipeline** outer replicates.
- Each replicate re-mixes one of three recorded WAV signals (`machinegun.wav`, `factory1.wav`, `factory2.wav`) into training and test examples, at nominal per-sequence **5 dB RMS SNR**, and re-trains the masked pretraining encoder. Within a dataset, budget and replicate the loss arms share the same corrupted data, unlabeled pretraining corpus, few-shot selection and seeded model initialization.
- The SSL corpus is built from **noisy UCR TRAIN sequences only**, tiled with a fresh small Gaussian jitter. UCR TEST feature arrays are separately corrupted for held-out evaluation and **are not used for pretraining**.
- The label-ID map is constructed from the **union of TRAIN and TEST label values**, rather than TRAIN labels alone. This accesses the *test class vocabulary*, not the test input features or targets as per-sample training examples. Without checking the original TSV class sets, do not assert that this mapping had zero protocol impact; future strictly train-only code should obtain the class vocabulary from TRAIN and reject unseen TEST labels.
- The WAV segments are chopped into sequence-length coordinate arrays and rescaled by per-sequence RMS; the code does not read and align WAV sample rates to UCR time-series sampling intervals. Hence this is a **recorded-waveform contamination stress test**, not validated sensor-specific physical noise injection. Neither all three source recordings nor the 5 dB mixture is independently demonstrated to be an impulsive-law model.

Recomputed equally weighted 16-cell mean test accuracies:

| Method | Equal-cell mean |
| --- | ---: |
| Scratch | **52.084%** |
| MSE | **58.392%** |
| L1 | **58.297%** |
| p=1.2 | **58.175%** |

Condition-wise leaders: scratch **1/16**, MSE **5/16**, L1 **8/16**, p=1.2 **2/16**. The p=1.2 equal-cell mean is **0.217 percentage points below MSE**, and when first averaging the two budgets *within each dataset*, it exceeds MSE on **4 of 8 datasets** and falls below it on **4 of 8**.

The 16 cells share eight underlying datasets, and test sizes/class balances differ widely. Their mean is a **macro-mean of accuracies at the dataset/budget level**, not a micro-average pooled over test samples or a probability that one optimizer is superior on a new dataset. The five replicate IDs share datasets and source recordings; do not run a naive 16- or 80-independent-trial significance calculation.

## 2A. Follow-up: NOISEX ranking stability across full-pipeline seeds and 8 dataset-level contrasts

**Archived-data follow-up, 2026-11-14 (no new GPU experiment).** The reported `8/16` L1 wins count is a **mean-over-five-seeds, dataset×shot** ranking and is correct. But a ranking of *means* does not imply that the same loss won most individual full-pipeline repetitions. All 16 means were re-expressed as integers using the actual per-dataset official test cardinalities saved in `results/noisex/run_metadata.json`, and each per-seed, per-loss ranking was determined within a **matched** dataset, shot budget and master seed. The test examples were reused across the loss arms within that replicate; ties in integer correct counts were **not** credited as strict wins.

- Exactly **7/16** of the per-condition mean winners were strict single-seed leaders on **at least three of the five** full-pipeline seeds. The remaining **9/16** were not strict-majority winners. Across all 16 cells, the distribution of how many seeds the mean winner strictly won was: `0:1, 1:5, 2:3, 3:5, 4:1, 5:1`.
- **ECG200, 5-shot:** The highest mean is **L1 = 68.8%** versus **p=1.2 = 68.4%**, a difference of **two correct predictions over five independent 100-example test evaluations**. Nevertheless L1 strictly wins **0/5** seedwise comparisons among the four methods; its superior mean arises from the magnitude of seedwise outcomes, not frequent individual wins. This is not a data inconsistency.
- **FaceAll, 10-shot:** L1 is the highest mean (**46.509%**) and strictly wins **all 5/5** seeds. Hence mean-winner stability genuinely varies by condition.

These calculations do **not** estimate how often the same method would win on a new independent dataset; each dataset provides only five coupled replicate outcomes, and testing four models for a maximum also creates selection effects.

### Independent full-pipeline replication is stronger than independent dataset count

The source's `derive_seed(master_seed,dataset,stage,substage)` and explicit `seed_all(...)` calls show that five master seeds each re-mix recorded noise, construct new augmented SSL data, initialize and train an encoder separately for each loss, sample matched few-shot indices and fine-tune. Within the same master seed, *losses* share the noisy train/test realization and pretrained corpus, and downstream initialisation seeds are aligned. These five replicates are valid for **within-dataset pipeline variability**. But two shot budgets and five runs per dataset still share the same underlying eight selected UCR datasets and official test-set records. Thus the evidence does **not** become 16 or 80 independent tasks.

To preserve dataset as the comparison unit, first average the 5-shot and 10-shot accuracies for each objective **within each dataset** (each has five complete pipeline replicate averages). Let `d_i=mean_{shots,seeds}(accuracy_{p=1.2}-accuracy_baseline)` for eight datasets, then enumerate all `2^8=256` sign assignments to the eight paired effects:

| p=1.2 minus | Equal-dataset effect (pp) | Dataset wins/losses | Two-sided signflip extreme/256 | Exploratory p |
| --- | ---: | ---: | ---: | ---: |
| MSE | **−0.21685** | 4/4 | **192/256** | **0.750000** |
| L1 | **−0.12243** | 3/5 | **202/256** | **0.7890625** |
| Scratch | **+6.09054** | 7/1 | **4/256** | **0.015625** |

A Holm step-down across the **two robust comparators** MSE and L1 makes both adjusted p-values **1.0**. The nominal scratch difference is not a clean measure of reconstruction-loss benefit: **scratch trains for 80 supervised epochs while pretrained approaches fine-tune for 50**, and shot conditions remain coupled. Do not make a post-hoc confirmatory pretraining-versus-scratch claim from this exploratory test.

The exact-allocation values are arithmetic facts of the archived dataset-level effect vector; their *formal* test interpretation requires paired-effect sign symmetry/exchangeability under an appropriate null. These are **eight deliberately selected datasets**, not a sample from a well-defined random population of industrial signal tasks. The pairings across full-pipeline seeds do not establish those cross-dataset assumptions. Lack of significance versus MSE/L1 does **not** establish loss equivalence.

**Disposition:** The submitted R1's L1 `8/16` win count remains correct; its conservative "mixed winners, no universal loss" conclusion is reinforced. For any future editor-facing passage, distinguish a winner **of five-seed means** from a reliable **per-seed majority** and limit the eight-dataset comparison accordingly. No change to frozen v70, R1 Table S2/S3, or experimental results is justified.

The existing `revision_audits/audit_external_cwru_noisex_evidence.py` was extended in place to recompute per-dataset correct counts, all 16 mean-winner strict-seed outcomes and the 256-way dataset-level contrasts. Its mathematical results were independently recomputed from live GitHub JSON. The updated Python script has **not** been executed within a full local repository checkout.

## 3. Manuscript disposition: preserve what is already correct

The unmarked assembled R1 main paper, Section 4.5 (PDF pages 85–86) and Discussion (page 87), already states:

- CWRU remained near chance, with 27.4–31.1% best means;
- NOISEX at 5 dB yielded L1 leading in 8/16 conditions, MSE 5/16, p-moment 2/16, scratch 1/16;
- CWRU is a diagnostic, and external rank choices do not supply a universal loss selector.

**Do not rewrite those correct values or use these limitations as a pretext for expanding 26SP into a bearing-diagnostics paper.**

If a future editable revision or reviewer response asks for CWRU experimental fairness, explicitly state:

> The CWRU holdout is load-disjoint at the source-file level, with 1,024-point windows and a stride of 512. Its 953 held-out windows therefore exhibit within-load overlap, and the absolute accuracies remain near four-class chance. Although few-shot samples and corruption realizations are shared across objective arms within each pipeline seed, the historical CWRU runner does not reset the optimizer/model-initialization and mini-batch-shuffle RNG streams separately for each loss. The resulting rankings should be considered descriptive and cannot isolate the reconstruction loss.

If a reviewer questions realism of NOISEX mixing, state:

> This experiment mixes randomly sampled segments of three recorded NOISEX audio waveforms at a nominal per-sequence 5 dB RMS ratio. Since the waveform samples are inserted without sample-rate or sensor-specific spectral matching to the UCR time axis, this experiment probes an additional recorded-noise perturbation protocol, not a physically calibrated industrial measurement condition.

Keep the protocol qualification concise and primarily in the Discussion or supplementary experiment description; the current article already discloses the near-chance negative finding.

## 4. Files and verification

Companion standard-library, no-download audit: `revision_audits/audit_external_cwru_noisex_evidence.py`. The above numerical results and source-control statements were independently reconstructed in this turn using the live archived GitHub files. The standalone Python checker has **not yet been executed from a complete local checkout**; no GPU experiments, noise-file downloads, data replacement, or modifications of frozen v70 were performed.

**Priority:** medium–high for CWRU loss-comparison fairness and absolute chance context; medium for NOISEX sampling-interval/label-catalog disclosures. No new experimental results warranted unless expressly demanded during review.
