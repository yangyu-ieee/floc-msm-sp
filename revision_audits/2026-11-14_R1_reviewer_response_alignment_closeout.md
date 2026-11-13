# 26SP R1: reviewer-response-to-clean-manuscript alignment and submission closeout

Audit date: 2026-11-14. Manuscript SIGPRO-D-26-02622, *Robust Masked Signal Modeling under Impulsive Noise: A Fractional Lower-Order Perspective*.

**Inputs inspected (read only):**
- Two separately named prior-file drafts: `RESPONSE_TO_REVIEWERS_DRAFT(1).md` (142 parsed lines, shorter draft) and `RESPONSE_TO_REVIEWERS_DRAFT.md` (190 parsed lines, longer checklist draft). Both begin as **drafts**; the longer file explicitly says author approval/Editorial Manager comparison remain open. Their relative authority is not established.
- `SIGPRO-D-26-02622_R1.pdf`, **clean/unmarked** manuscript at assembled PDF pages 65–87 (printed main-text pages 1–23). In particular PDF page 78 = printed Table 1 p.14; PDF page 84 = Table 3 p.20; printed p.19–20 = mask-pattern subsection.
- Frozen `revision_v70_20261010/results/main_table_reconciliation_20261009/results.json` (explicit designated Table 1 source), `results/masking/random_token_exact/results.json` and `results/masking/contiguous_block_v2/results.json`.
- Prior audit `revision_audits/2026-11-14_R1_supplement_and_public_gate_audit.md`: Table 1 and Table 3 clean-R1 numerical cells validated, but full Supplementary Tables S1–S18 not independently transcribed due unavailable supplemental source.

No Library draft was overwritten; no frozen archive, submitted manuscript or editorial record was changed. **Neither existing response draft is cleared for author submission.** This audit does not establish which letter, if any, is already in the editorial portal.

## Priority P0 — definite numeric contradictions in BOTH drafts

| Response material | Stale author-response wording, both draft files | Clean R1 and frozen designated evidence | Severity |
| --- | --- | --- | --- |
| Proactive sampler correction | MSE `94.3/95.0/95.8/97.3%`; "FLOC" p=1.2 `94.4/95.9/96.7/98.1%` | Table 1 MSE **93.6/94.5/95.7/97.0%**, FLOM-based p-moment p=1.2 **94.1/96.0/97.0/98.5%** at n=50/100/200/500 | **P0** |
| Claimed effect range | "corrected FLOC–MSE mean gaps **0.1–1.0** pp" in response to R1.1, and old numbers in proactive disclosure | Actual raw five-seed mean differences **+0.44/+1.52/+1.24/+1.44 pp**; R1 Abstract/Intro/Main say **0.4–1.5 pp** | **P0** |
| Largest mean by budget | "L1 has highest mean at three of four budgets" | p-moment p=1.2 leads **n=50/200**, L1 leads **n=100/500**, i.e. two each | **P0** |
| Reviewer 3, masking | "random-token masking has higher mean accuracy in all eight displayed cells, by **0.04–1.68** pp" (both R3.3 responses) | Seven positive displayed differences and one **0.00 pp** displayed tie at p=1.2/n=200, max **1.60 pp**; clean R1 §4.4 says **7 of 8 / one tie / 0–1.60 pp** | **P0** |

The corrected Table 1 source was independently re-read in this turn, and the four unrounded p=1.2-minus-MSE gaps were recomputed from archived means. The masking differences were independently recomputed from both archived strategy JSONs; at n=200/p=1.2, the raw floating-point discrepancy is only about `0.0000012` pp and rounds to **0.00 pp**, as declared by the R1 table/claim. **Use table-display precision** for the tie rather than claim strong numerical equality at arbitrary machine precision.

**Do not** change Table 1, retune hyperparameters, or substitute older `main_alpha15` values to make the response appear consistent. Update only editable review-response prose after verifying which file is the intended author-controlled final version.

## Priority P1 — obsolete main-manuscript page references / terminology

Both drafts' location maps are from an older, shorter version. The **clean R1 main text has 23 printed pages** and is embedded at assembled PDF pages 65–87, so claims such as "Discussion p.16" or "Fig.1 p.8" are now wrong.

**Verified printed-page map from the clean R1:**

| Content / recommended response locator | Actual printed R1 main pages |
| --- | --- |
| Abstract, Introduction | pp. 1–3 |
| Related Work | pp. 4–5 |
| Model/objectives/theory | pp. 6–10 |
| Experimental protocol and architecture | pp. 11–12 |
| Workflow **Figure 1** | p. **13** |
| Corrected core **Table 1** | p. **14** |
| Actual-residual/score/gradient diagnostics, crossover, local slices | pp. **14–17**; **Figure 2 p.17** |
| Joint corruption-and-scaling / robust baselines / masking | pp. **18–20**; **Table 2 p.19; Table 3 p.20** |
| Student-t, multi-α, randomized task, UCR/CWRU/NOISEX discussion | pp. **20–22** |
| **Discussion** | pp. **22–23** |
| **Conclusion** | p. **23** |

Only the **main-manuscript** page locations above were rechecked. The draft references to Supplementary Notes/Tables S1–S18 are **not all independently transcribed** against a final supplementary artifact; do not label those references fully verified without seeing the exact supplemental PDF/TeX that will accompany the submission.

The revised main (§3.1, PDF p.71; §3.2, p.72; Table 1, PDF p.78) consistently uses **"FLOM-based p-moment loss" / "p-moment loss"** for `|r|^p` and reserves **"FLOC"** for *fractional lower-order covariance*. Both drafts nevertheless use "FLOC" repeatedly in the authors' own descriptions of their pointwise loss. Update **author-written response text**, while leaving *verbatim reviewer quotations* and their original wording untouched.

One more claim-to-location inconsistency occurs in the longer draft R3.2: it says **the Conclusion now directly states that there is no reliable residual-tail estimate for p selection**. In the actual clean R1, §4.3 (printed p.15) says the Hill estimate has no stable plateau and `we do not estimate α to select p`, while the Conclusion (printed p.23) focuses on non-universality and no convergence guarantee. **Attribute this specific statement to §4.3, not literally to the Conclusion.**

## Paste-ready minimal English replacement passages

### A. Proactive sampler disclosure — replace only the stale numerical/results paragraph

> We regenerated the affected corrected-CMS experiments and replaced the corresponding manuscript results. In the designated corrected five-seed α=1.5 core benchmark (Table 1), mean test accuracy at 50/100/200/500 total labeled examples is 93.6/94.5/95.7/97.0% for MSE and 94.1/96.0/97.0/98.5% for the FLOM-based p-moment loss (p=1.2). The corresponding unrounded mean differences are +0.44/+1.52/+1.24/+1.44 percentage points. The highest mean alternates between the p-moment loss (50 and 200 labels) and L1 (100 and 500 labels). Each loss has one pretrained encoder evaluated over five matched downstream fine-tuning seeds; the minimum attainable two-sided exact sign-flip p-value is 0.0625. We therefore report these gains descriptively rather than as established significant differences. The incorrect original α=1.5 results and significance claims are not retained as evidence, and the revision is not a matched rerun of the original experiment.

Keep the subsequent correct remarks on seven sampler tests and corrected Cauchy independent-stream Welch analysis, as supported by their separate archived audits.

### B. Reviewer 1.1 novelty — replace the stale effect-size comparison sentence

> In the corrected five-seed core trial, the p=1.2 p-moment loss exceeds MSE in mean by 0.44–1.52 percentage points, without a statistically established five-seed advantage. A separate joint corruption-and-scaling protocol audit exhibits within-audit accuracy contrasts of 1.9–3.6 points; because its conditions and checkpoints differ from those of the core benchmark, this comparison does not isolate normalization or establish a general causal ranking of preprocessing versus loss effects. Our measured raw network-gradient norms also do not follow residual-score concentration. These observations motivate a protocol-scoped study, not a universal loss choice or new stable-law theorem.

### C. Reviewer 3.3 masking — replace the incorrect 8/8 / range sentence

> Under the common nominal 50% token-mask ratio, exact-count random-token masking achieves higher mean accuracy in seven of eight loss–budget cells and ties the p-moment/200-label cell at the displayed precision; observed differences range from 0.00 to 1.60 percentage points (Supplementary Table S5). Because Conv1D token receptive fields overlap, a fixed token-mask fraction does not ensure matched direct raw-sample visibility across random and contiguous-block patterns. The separate 25%/75% endpoint sensitivity runs use Bernoulli token masking and do not identify an optimal mask ratio. We therefore present these masking comparisons descriptively rather than as isolation of a pure pattern effect.

### D. Reviewer 3.2 residual-tail / convergence sentence — fix the attribution

> Section 4.3 reports that threshold-sensitive Hill diagnostics do not provide a stable residual-tail-index plateau, so we do not estimate the learned residual α to select p. The Discussion and Conclusion separately state that the observed finite-run diagnostics do not yield a universal loss-selection rule or a convergence guarantee.

## Pre-submission acceptance checklist

- **P0**: one author-approved response letter is selected; every copied Table 1 number, primary effect range, winner count, masking win count and range is changed consistently in **every** occurrence (not just the first paragraph).
- **P1**: the location map refers to **printed pages** of the actual 23-page clean R1, not assembled PDF file page numbers nor old 17-page drafts.
- **P1**: replace misuse of FLOC in authors' prose with FLOM-based p-moment loss; retain exact reviewer-quote wording.
- **Open**: verify every Supplementary Note/Table/Figure locator against the actual **final submitted supplementary artifact**, not merely a historical script's anticipated labels. Also verify editor submission portal status and author approval separately.

**Final disposition:** The primary manuscript results checked here are internally supported; the **reviewer-response drafts are not synchronized with those results** and should not be filed as-is. This is a higher-priority delivery defect than additional post-hoc experiment discovery. No further GPU computation is required for these corrections.

## Implementation closeout: full candidate created, legacy drafts preserved

**Candidate file:** `revision_audits/2026-11-14_R1_response_to_reviewers_reconciled_CANDIDATE.md` (created 2026-11-14 from the shorter `RESPONSE_TO_REVIEWERS_DRAFT(1).md` Library version; separately committed as `4043416`).

**Completed in this new candidate, not in the original Library drafts:**
- Replaced all outdated corrected-CMS primary MSE/p=1.2 figures, four absolute effects, and the incorrect 3/4 L1 win claim with the designated Table 1 values and 2/4–2/4 budget split.
- Replaced both the R1.1 historical `0.1–1.0` claim and the R3.3 incorrect `8/8, 0.04–1.68` masking claim with `0.4–1.5` primary effect range and `7/8 plus a displayed tie, 0.00–1.60` masking range.
- Updated all **11** main-manuscript `**Location:**` lines to printed pages of the verified **23-page** clean R1.
- Updated the authors' references to the pointwise reconstruction objective to FLOM-based `p`-moment-loss wording; **all verbatim reviewer quote lines remain byte-for-byte identical** to the source draft. The FLOS/FLOC literature reference remains FLOC because that is mathematically a *covariance* discussion.
- Corrected R3.2's false attribution of the residual-tail-index statement to the Conclusion: it is in §4.3, with more general limitations in Discussion/Conclusion.
- Corrected the response's inaccurate description of Figure 1's label-budget annotation: the actual diagram says only `few-label`; the 50–500 total-label budgets appear in the main protocol, not in the diagram itself. The response now distinguishes those from per-class external shot budgets.
- Preserved reviewer 1 and reviewer 3 numbering, all original quoted comments, and the 13 subsection headings. No substantive experimental claims were added beyond source-grounded clarifications.
- Added an invisible Markdown author-review status note near the top of the new candidate, with **no** assertion that it is the final, approved or uploaded editorial response.

**Verified after write:** new candidate re-fetched from GitHub; the reviewer-quote lines and section headings were compared directly against the original shorter draft, all 11 revised main `**Location:**` lines were inspected, and stale Table 1/ masking values were absent from the author-response text. The longer `RESPONSE_TO_REVIEWERS_DRAFT.md` remains unchanged and stale.

**Still open:** author's choice and approval of which reply is canonical; whether the exact Supplementary Tables S1–S18 / Notes / figures correspond to the finally submitted supplement; whether any response has already been uploaded in Editorial Manager; and all author-side submission formatting/metadata checks. The GitHub Markdown candidate is a working file, **not** a submission action, and the archived R1 / frozen v70 have not been touched.

## Three-source recheck and Editorial Manager assembly boundary (2026-11-14)

A broader search uncovered a **third independent author-side response text** in the user's document Library: `粘贴的文本 (1)(4).txt` (143 lines of retrieved text). Unlike the two stale Markdown drafts described at the start of this report, this copy **already has the current 0.4–1.5 pp core-result range, 7/8 plus one masking tie, and updated 23-page main-text locations**. It also contains additional source/protocol qualifications: the historical L1/p=1 duplicate-objective inconsistency, Bernoulli versus exact-count diagnostic masking, the missing historical source for one random-token masking run, and the noncanonical full-signal reconstruction target relative to masked-patch-only MAE. The Library title is generic and no author approval is recorded; **its chronology or authority relative to the two named Markdown drafts cannot be determined**. It would be inaccurate to say every previously existing reply version was stale.

The GitHub **complete reply candidate** at `revision_audits/2026-11-14_R1_response_to_reviewers_reconciled_CANDIDATE.md` was updated (commit `e3c92e6`) to incorporate these source-backed protocol disclosures **without** replacing correct core data, reverting to old page numbers, or silently adopting unverified Supplementary Material page locators. The underlying original Library files were **not overwritten**. The candidate deliberately uses printed main-paper page numbers and supplementary table/note numbers, rather than asserting exact supplement page numbers not independently available here.

**Reviewer response coverage check:** The candidate has all **five** numbered Reviewer 1 replies (R1.1–R1.5) and all **six** Reviewer 3 replies (R3.1–R3.6). Its `>` reviewer-quotation lines exactly match both the shorter Markdown draft and the separate plain-text copy; their shared provenance does not independently authenticate Reviewer 1 comments. Against the supplied **original Reviewer 3 DOCX**, all **six** numbered comments match after whitespace normalization; the only notation normalization is the DOCX's LaTeX `$\\alpha$` in item 2 versus a printed Greek `α` in the candidate. The DOCX general assessment is included as well. The available author-side copies do not have a Reviewer 2 section; **absence of R2 from those copies is not proof that no additional editor/reviewer instruction exists**, because the full decision/portal response inventory was not retrieved.

**Editorial-system artifact evidence:** the first page of `SIGPRO-D-26-02622_R1.pdf` explicitly displays `SIGPRO-D-26-02622R1` and the footer `Powered by Editorial Manager® and ProduXion Manager®`. Assembled PDF page 3 has a *Supplementary material* item pointing to `Supplementary_Material_v73.pdf`; page 4 contains the revision cover letter and page 5 the Highlights. This is stronger evidence than an ordinary local-compiled author PDF that an R1 submission-package preview/assembly exists. **It does not independently establish that the submission workflow was finalized, which response-file version was uploaded, or that the linked Supplementary PDF's full pages are included in the assembled 94-page PDF.** A filename/link page is not sufficient to transcribe Tables S1–S18, verify source hashes, or claim a supplementary numerical pass.

**Actionable terminal gate:** The existing candidate is now a suitable **review starting point**. To call it author-final, (i) identify the canonical response file from the actual portal or author, (ii) independently open the exact `Supplementary_Material_v73.pdf` or later submitted artifact and verify all cited S-table/figure/note locators, and (iii) confirm submission status in the journal portal. No research rerun is implied or justified solely by this editorial-source uncertainty.
