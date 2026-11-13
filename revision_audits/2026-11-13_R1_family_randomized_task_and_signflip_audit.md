# 26SP R1: randomized-family task boundary, seedwise ranking resolution and float32 sign-flip tie

Date: 2026-11-13. Scope: **audit only**. Submitted R1, frozen `revision_v70_20261010` code, all five recorded runs, their summaries and the compiled manuscript remain unchanged.

## 1. Source and archive provenance

Checked:

- `revision_v70_20261010/code/floc_msm_randomized_class_audit.py` (`gen_unlabeled`, `gen_cls`, exact-count masking, pretraining reset, downstream reset).
- `revision_v70_20261010/code/analyze_family_randomized.py` (archived 5-seed aggregation and sign-flip enumeration).
- Five `revision_v70_20261010/results/family_randomized/seed_{0,1,2,3,4}/{config.json,results.json}` pairs.
- `revision_v70_20261010/results/family_randomized_summary.json` and `family_randomized_results.md`.
- Unmarked submitted R1 Section 4.5 / Supplementary Table S17; the manuscript **already calls** the two tasks distinct protocols and makes only a descriptive cross-task comparison.

All five run configurations record the same source SHA-256 `e6b3517e3c66e564142d50b8eef66ca5a23683312c800c7c2370073d0e4272c3`, `class_mode=family_randomized`, exact-count `random_token` masking (16 of 32), the corrected CMS alpha=1.5 generator, 20 SSL epochs, `pt_samples=10000`, and `n_seeds=1`. The seed IDs for the five separately invoked complete runs are 0–4.

## 2. The randomized task changes the meaning of the target labels, not just their parameters

The *fixed-template primary benchmark* and the *randomized-family diagnostic* have different conditional signal definitions. In the shared generator, `gen_cls` branches on `CONFIG['class_mode']`:

| Class | Fixed-template task | Randomized-family task |
| --- | --- | --- |
| 0 | fixed two-sinusoid frequencies/amplitudes | two sinusoids with example-wise randomized frequencies |
| 1 | fixed two-frequency sinusoid | randomized linear-frequency chirp |
| 2 | square wave (sign of sinusoid) | randomized sparse pulse train |
| 3 | sawtooth / ramp waveform | randomized exponentially decaying sinusoid |
| 4 | fixed pulse train | random walk |

The randomized task's **five** label families intentionally match the five *unlabeled pretraining* waveform families, whereas fixed-template class IDs do not have the same semantics. In particular, family 2, 3 and 4 target classes are qualitatively **different classes**, not simply parameter-perturbed versions of the fixed template with the same index.

Consequences:

1. A drop from the fixed-template accuracies (roughly 92.5–98.6% for the four compared pretrained losses) to randomized-family accuracies (roughly 36.7–67.0%) **does not measure** the performance drop of a fixed classifier under a parameter shift. Models are separately trained and even the conditional target class definitions differ.
2. In randomized-family mode the training and test generators use the **same family-specific parameter distributions**, drawing parameters independently *per example*. The family identities are observed in the unlabeled pretraining mixture. This probes *in-distribution classification with varying waveform parameters*, not held-out parameter **ranges**, unseen families, or OOD generalization.
3. The pretraining corpus is generated as **unlabeled** mixtures of those waveform families. It is not given labels or test arrays. Seeing the same family types during SSL is deliberate alignment, not by itself data leakage.
4. Labels are generated as `cls=i%5`. All training budgets are divisible by five, so they contain exactly 10/20/40/100 examples **per class** for total budgets 50/100/200/500; the 500-example test set has exactly 100 examples per class (chance 20%). The budget quantity in Tables is *total labels*, not labels per class.

## 3. Within-run controls, repeated pretraining and the learning-curve boundary

Unlike the primary corrected main table's single SSL checkpoint per loss, this protocol launches **five separate complete runs**, each regenerating the unlabeled pretraining dataset, re-pretraining one encoder per loss with the same run-specific initialization (`torch.manual_seed(args.seed+1)`) and loader seed (`args.seed+2`), and drawing its own downstream training/test data. For every seed and label budget, `torch.manual_seed(run_seed); np.random.seed(run_seed)` resets the downstream streams **before every loss**, so at that one budget the training and testing realizations are matched across losses, including scratch.

Each of the five full-pipeline seeds produces **one downstream evaluation** per pretrained checkpoint/loss/budget. Its five-seed variability therefore includes SSL initialization/data variation *and* downstream data/training variation, but does not separately estimate either component.

Across total label budgets, a repeated seed produces prefix-nested *training* examples (the sampler draws examples sequentially with class cycling) but **different test sets**, because the test generator consumes the same NumPy stream *after* the budget-dependent number of training examples. Moreover, pretrained cases with 500 labels get 50 fine-tuning epochs, vs 30 at 50/100/200, while scratch always gets 80. Therefore cross-budget numbers do not isolate training set size or optimizer budget.

## 4. Exact integer-count recovery of the best-mean margins

Each of the five archived runs reports accuracy on 500 test examples. Recover the **integer** number of correct classifications via `round(500 * accuracy)` for each run before aggregating: all 120 loss × budget × seed cells are consistent with a multiple of 1/500.

| Total labeled examples | Winner | Runner-up | Winner mean | Runner-up mean | Margin (pp) | Additional correct out of 2,500 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 50 | Charbonnier | p=1.2 | 39.48% | 39.28% | **0.20** | **5** |
| 100 | MSE | Charbonnier | 48.16% | 47.92% | **0.24** | **6** |
| 200 | L1 | p=1.2 | 57.60% | 56.64% | **0.96** | **24** |
| 500 | Charbonnier | L1 | 67.00% | 66.96% | **0.04** | **1** |

Winning loss names agree with the submitted manuscript's descriptive statement. At 50 labels, the mean leader is first in only **one of the five individual seeds**; at 100 labels MSE is first in only **one** seed; at 500 labels Charbonnier tops **three** seeds. These are selected post hoc and are not confidence statements about the population-optimal loss. A 0.04 percentage-point edge has essentially no empirical ranking resolution with five 500-example test sets.

For p=1.2 versus MSE, the recorded paired mean differences are **+0.36, −1.56, +3.40 and +2.68 pp** at budgets 50/100/200/500, with exact five-pair two-sided integer-count sign-flip values **0.9375, 0.7500, 0.0625, 0.0625**. A p-value of 0.0625 is the smallest nonzero two-sided p available from five consistently signed pairs, and these exploratory contrasts do not claim confirmatory significance. Different budgets reuse the same five full-pipeline seed IDs; do not pool them into 20 independent trials.

## 5. A real but small arithmetic discrepancy in the archived p-value

In `analyze_family_randomized.py`, the enumerated sign-flip test uses **float32-derived accuracy fractions** and declares an assignment extreme when its absolute statistic is at least `observed - 1e-15`.

For the **Charbonnier versus MSE, 50-total-label** contrast, the exact integer correct-count differences per seed are

`[-4, -2, +45, +4, -29]` (out of 500 each), summing to `+14`.

Enumeration of all 32 sign flips on this integer vector yields **28/32 = 0.8750**. The archived `family_randomized_summary.json` reports **26/32 = 0.8125**. The origin is float32 rounding: two signed permutations with exact absolute sum 14 are represented by floating sums approximately `0.0279999673`, compared with observed `0.0280000269`; their absolute discrepancy is only **5.96×10^-8**. The archive's `1e-15` tie tolerance is far too strict at this accuracy precision.

The independent audit recomputed **all 20 non-MSE-loss × budget five-pair sign-flip contrasts** from integer correct counts. **This is the only mismatch** against the archived p-value summary; all archived means, standard deviations, classification accuracies and best-mean rankings reconcile.

Disposition: record **p=0.8750** as the integer-count-corrected *audit value*. Do **not silently overwrite** the frozen historical summary or submitted R1. Neither 0.8125 nor 0.8750 suggests significance, and this is not evidence of intentional test manipulation.

## 6. Scratch baseline at 100 labels and the limit of pretraining conclusions

At budget 100 the *scratch* model scores exactly **20.00%** on two of the five seeds (0 and 2), i.e. **100/500 correct**, equal to uniform chance in the balanced five-class test. Its five-seed mean is **29.48%**, versus **48.16%** for MSE pretraining, a +18.68 pp mean contrast. The two chance-level scratch seeds contribute **10.04 pp** of that 18.68 pp gain (about **53.75%**).

This is a **transparent descriptive decomposition**, not a finding of a specific model-collapse mechanism: the archived family-randomized runs lack prediction histograms/confusion matrices needed to tell whether these seeds collapsed to a single class or produced chance-level accuracy for some other reason. Scratch also uses 80 downstream epochs vs the pretrained model's 30 at this budget. Do not interpret the mean improvement as a controlled attribution to pretraining under equal optimization budgets; on the other hand, scratch received **more**, not fewer, epochs.

## 7. Manuscript disposition

The current submitted R1 Section 4.5 properly describes a different task and does not inferentially compare its accuracies with the fixed-template table. There is **no need** to revise the core accuracy results or to collect GPU data.

If a future revision asks specifically about generalization, the highest-value compact clarification is:

> The randomized-family evaluation is a separate in-distribution classification task: its five class definitions correspond to the five unlabeled pretraining waveform families, with parameters independently redrawn for each labeled and test example. The class definitions differ from those of the fixed-template benchmark, so cross-task accuracy differences do not estimate a fixed-model distribution-shift penalty. Each of five pipeline seeds independently retrains the encoder; the identity of the highest-mean objective is descriptive, with differences as small as 0.04 percentage points.

Do not put the float32 p discrepancy into the main narrative unless the full supplementary sign-flip table or an updated analysis script is being distributed. If statistical values are revised, use integer correct-count tests and document the historical value side-by-side.

Companion standard-library checker: `revision_audits/audit_family_randomized_task_scope.py`. All cited summary and sign-flip values were independently recalculated from GitHub's live archived JSON; the newly committed Python checker itself has **not yet run in a complete local repo checkout**. Frozen v70 and submitted R1 unchanged.
