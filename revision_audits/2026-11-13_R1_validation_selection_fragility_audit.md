# 26SP R1 validation-selection provenance and near-ties

Audit date: 2026-11-13. Status: prospective amendments only. **Do not alter the submitted R1 PDF, v70 results, or historical runner.**

## Checked artifacts

- `revision_v70_20261010/code/r2_mcc_baseline.py` (actual selection runner, rather than the nonexistent `validation_selected_baselines_corrected.py`).
- `revision_v70_20261010/results/validation_selected/{config.json,results.json,validation_selection.json}` (budgets 50, 200, 500).
- `revision_v70_20261010/results/validation_selected_n100/{config.json,results.json,validation_selection.json}` (budget 100).
- Unmarked R1 manuscript, Table 3 and Section 4.4 (assembled PDF page 84): validation-only selection, Bernoulli(0.5) masking; explicitly independent of the exact-count main trial.

## 1. No code evidence of held-out test selection

The runner sets `Xtr,ytr=gen_cls(n_train,...)`, then `Xval,yval=gen_cls(500,...)`, and separately `Xte,yte=gen_cls(500,...)` before training. Per loss, budget and downstream seed it resets both NumPy and PyTorch to the same seed, and therefore matches validation/test draws across candidate losses within the same budget and seed. The `selection` loop at lines 269–288 selects the highest **mean validation classification accuracy over five seeds**; `test_mean` enters only the archived report, *not* the selection key.

This is a valid validation-only model-selection design under the recorded protocol, not evidence of test leakage. The four budgets are assembled from two runs, but both archived configs use the same runner SHA-256 `241517f9c085d45ae71ab74396750b5f496797b834e5d169e88d9725a7852261`, master seed, architecture and hyperparameter list. The separate n=100 invocation is an important audit detail.

## 2. Selection can hinge on zero or one correct validation prediction

Each five-seed candidate uses 500 validation examples per seed, so each candidate has 2,500 validation predictions in total. Reconstruct each integer correct count as `round(500 * val_seed_accuracy)`, not by naive direct comparisons of float32 mean values.

| Labeled budget | Family | Archived selected variant | Total-correct lead over runner-up (out of 2500) | Interpretation |
| ---: | --- | --- | ---: | --- |
| 50 | Huber | delta=0.5 | 13 | Numerically distinct validation leader |
| 50 | Charbonnier | epsilon=0.001 | 31 | Numerically distinct validation leader |
| 50 | MCC | sigma=0.5 | 43 | Numerically distinct validation leader |
| 100 | Huber | delta=0.25 | 20 | Numerically distinct validation leader |
| 100 | Charbonnier | epsilon=0.01 | **0** | **Tie** with epsilon=0.001 in integer correct count; raw mean differs by only ~1.19e-8 accuracy fraction, a floating-point artifact |
| 100 | MCC | sigma=0.5 | 3 | Three-correct-prediction margin |
| 200 | Huber | delta=0.5 | **1** | 0.04 percentage point margin over delta=0.25 |
| 200 | Charbonnier | epsilon=0.001 | 8 | Eight-correct-prediction margin |
| 200 | MCC | sigma=0.25 | **0** | **Exact tie** with sigma=0.5; first candidate in listed order is returned |
| 500 | Huber | delta=0.25 | 15 | Numerically distinct validation leader |
| 500 | Charbonnier | epsilon=0.01 | **1** | 0.04 pp validation margin over epsilon=0.001 |
| 500 | MCC | sigma=0.25 | **1** | 0.04 pp validation margin over sigma=0.1 |

These are *within-family hyperparameter-selection gaps*, not significance tests and not evidence of an error in the held-out test accuracy.

### Additional stress evidence: held-out test rankings sometimes differ

- At 500 labels, Charbonnier `epsilon=0.01` is selected by one extra validation correct prediction; its archived test mean is **97.96%**, while the nonselected `epsilon=0.001` reaches **98.52%** in the same five-seed archive.
- At 500 labels, MCC `sigma=0.25` is selected by one extra validation correct prediction; its archived test mean is **97.64%**, while nonselected `sigma=0.1` reaches **98.24%**.
- These are *diagnostics of near-tie selection instability*. **Never replace the test values in Table 3 with the better retrospective test candidate**: that would be invalid test-driven tuning and break the recorded procedure.

## 3. Implications for paper presentation

Table 3 is a valid protocol-specific sensitivity comparison with recorded candidate grids, not a global optimum-finding result. The selected Huber, Charbonnier, and MCC hyperparameters are allowed to differ by label budget because the archived selection rule explicitly did so; it would be wrong to imply one globally tuned setting was used across all budgets.

Suggested Section 4.4 addition for a *future editable R1*:

> Hyperparameters for Huber, Charbonnier and MCC were selected separately within each label budget by the mean validation accuracy over five matched downstream trials, using 500 validation examples per trial; held-out test accuracy was not used for selection. Some chosen settings were effectively tied at the resolution of the validation results: two family–budget selections had identical total validation correct counts, and three others exceeded the runner-up by only one correct prediction across the 2,500 pooled validation predictions. Accordingly, selected parameter values and small test-mean differences should be read as budget- and split-dependent sensitivity outcomes, not stable optimal hyperparameters or statistically significant rankings.

Suggested Supplementary Table S12 note:

> Selection ties are identified from integer correct counts reconstructed from the five stored validation seed accuracies. Where the archived floating-point selection rule broke an integer-count tie, we retain its historically selected variant and test outcome without retroactive test-based re-selection.

Do not extend this wording to every baseline: MSE, L1, and fixed p=1.2 have no comparable grid search in this audit. The fixed-parameter core Table 1 and validation-selected Table 3 are separate, with different masking protocols; they must not be pooled.

## 3A. Retrospective evaluation ranking and independent p=1.2 comparator (additional audit)

This is a strictly **post-hoc consistency assessment** of the existing held-out test outcomes, not a change to the selection rule. The actual validation-only selected parameters above are always frozen **before** looking at these tests. The code resets NumPy/Torch to each same downstream seed for the candidates and fixed p=1.2, and `eval_downstream()` draws `Xtr,ytr`, `Xval,yval`, and `Xte,yte` **sequentially** from the same generator with no training on validation or test arrays. In particular, candidate test sets and validation sets are matched **across loss alternatives** at a given seed and label budget, but are different realizations from one another.

Exactly 12 loss-family × budget selections are recorded. Reconstructing all candidate accuracies as integer correct-prediction counts out of 500 per seed (five seeds per budget) gives:

- **9/12** validation-selected variants are **unique test-mean winners** within their own loss family; at `n=100` Charbonnier `epsilon=.01` **ties** `epsilon=.001` in held-out mean, giving a total of **10/12 selections at or tied for the family’s retrospective best test mean**.
- The **two exceptions** both occur at `n=500`. Charbonnier selects `epsilon=.01` by a 1/2500 validation-correct advantage over `epsilon=.001`, but its observed test mean is **97.96%** versus **98.52%** (14 fewer correct/2500, −0.56 pp). MCC selects `sigma=.25` by 1/2500 over `sigma=.1`, but test mean is **97.64%** versus **98.24%** (15 fewer correct/2500, −0.60 pp).
- The mean hindsight test gap to the best variant in each family × budget is **0.0967 pp** across these 12 conditions; 10 are zero. This quantity is **retrospective and optimistic by definition for the hindsight oracle**, not a valid alternative procedure and not an estimate of generalization regret in new datasets.
- Selected validation-minus-test accuracy, averaged equally over the 12 archived cells, is **−0.1767 pp**. Thus this finite archive does **not** display uniform positive validation-to-test optimism for the chosen variants. This does not establish absence of tuning-induced selection bias in general; the 12 cells share seeds, data-generating mechanisms and repeated pipelines.

### Fixed p=1.2 against validation-selected baselines, within the same Bernoulli-mask protocol

The prior §2 examined **only within-family tuning gaps**. This additional calculation compares the **precommitted fixed p=1.2 checkpoint** to each *validation-chosen* Huber/Charbonnier/MCC checkpoint using the same seed-specific held-out test realizations, not the exact-count-mask primary Table 1.

Each cell is five-seed mean accuracy difference in **percentage points**:

| Total labels | p=1.2 − selected Huber | p=1.2 − selected Charbonnier | p=1.2 − selected MCC |
| ---: | ---: | ---: | ---: |
| 50 | +0.40 | −0.72 | +0.68 |
| 100 | −0.16 | −0.32 | 0.00 |
| 200 | +0.20 | −0.84 | −0.64 |
| 500 | −0.24 | +0.16 | +0.48 |

The fixed-p method wins **5/12**, loses **6/12**, and ties **1/12** in mean, without any pooled cross-budget significance inference. Enumerating the 32 integer-correct-count sign flips per five-seed contrast yields two-sided p-values **no smaller than 0.125** across the 12 cells. These are exploratory, post-selection conditional contrasts; they do **not** prove formal equivalence, and the small seed count limits inference.

Crucially, the validation-selected candidates received **within-family hyperparameter search**, while fixed p=1.2 did not receive a matched p grid search within this specific run. The comparison answers whether one chosen fixed p beats independently selected robust alternatives under **this one Bernoulli-mask protocol**, not what an optimally tuned p-moment family can achieve.

**Implication:** Test/validation isolation appears correctly coded; near-tie selection can change test ordering by around half a point; there is no evidence from this audit that p=1.2 universally dominates validation-selected robust baselines. The clean R1 Table 3 already calls this a separate sensitivity protocol, so there is no need to revise or reselect its reported test means, and no necessity for GPU reruns.

The existing `revision_audits/audit_validation_selection_ties.py` checker has been extended *in place* with the retrospective test rank, paired p=1.2 test contrasts and 12-cell summary. All numbers were independently recomputed from the frozen GitHub JSON via integer counts; the updated Python checker itself has **not been executed in a full repository checkout**.

## 4. Verification and disposition

Data audit: independent recomputation of per-seed integer validation correct counts from all 12 family/budget conditions on the live archived GitHub data. The selected variants match the stored runner rule. No training was performed and no experimental result was modified.

Standard-library reproducibility check: `revision_audits/audit_validation_selection_ties.py`. It reconstructs per-candidate validation correct counts and flags the two ties plus three single-count margins. The newly committed script itself has not been run inside a full local clone; its logic has been cross-checked on retrieved JSON values.

**Priority:** medium to high for accurate interpretation and disclosure; not a reason to withdraw any existing Table 3 numbers. Avoid overcomplicating the submitted revision with new tuning experiments unless a reviewer requires them.
