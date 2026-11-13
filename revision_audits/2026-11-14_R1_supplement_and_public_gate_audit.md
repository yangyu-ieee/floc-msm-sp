# SIGPRO-D-26-02622 R1: supplementary-source dependency preflight and public numerical gate

Audit date: 2026-11-14. Scope: the currently frozen `revision_v70_20261010/` **GitHub** release, the assembled **clean** `SIGPRO-D-26-02622_R1.pdf`, and archived JSON. This is a post-assembly, read-only artifact audit, **not** evidence of author approval or Editorial Manager submission and **not** a full Supplementary Materials verification. No GPU training, archive edits or manuscript changes.

## 1. Confirmed source-availability boundary

Two archived scripts are source-dependent:

- `code/audit_supplement_tables.py` reads `manuscript/sp_main.tex`, a supplementary TeX candidate (`manuscript/sp_supplementary.tex` or `supplement/sp_supplementary.tex`), separate `manuscript/ucr_cms_dataset_table.tex` / `manuscript/noisex92_dataset_table.tex` table fragments, and an epoch-20 NPZ score sample for residual-tail Table S14. The supplementary path selection uses `next(... if path.exists())`: with both TeX candidates absent from this release, **the script raises StopIteration at module initialization**, before conducting any table check.
- `code/audit_release.py` attempts `manuscript/sp_main.tex`, `supplement/sp_supplementary.tex`, and `SOURCE_PROVENANCE.json`, none of which is present in the frozen public tree. It therefore cannot be represented as an executable whole-release gate for this distribution.

As directly confirmed against the GitHub tree, **all these paths are absent**:

```
revision_v70_20261010/manuscript/sp_main.tex
revision_v70_20261010/manuscript/sp_supplementary.tex
revision_v70_20261010/supplement/sp_supplementary.tex
revision_v70_20261010/manuscript/ucr_cms_dataset_table.tex
revision_v70_20261010/manuscript/noisex92_dataset_table.tex
revision_v70_20261010/SOURCE_PROVENANCE.json
revision_v70_20261010/results/diagnostics/epoch20_score_samples.npz
```

The small derived `results/diagnostics/tail_index_sensitivity.json` *is* present; it is not a substitute for the NPZ needed to verify the recorded source SHA. The versioned tree does include `code/audit_supplement_tables.py`, `code/audit_release.py`, `code/run_main_table_reconciliation.py`, the primary JSON results, `paper_tables/diagnostic_table_snapshots.tex` (selected snapshots, **not** the full manuscript), and `MANIFEST_SHA256.txt`.

**This is a package/dependency completeness limitation, not a discovered paper-number discrepancy.** The frozen README explicitly discloses that some audit scripts need private manuscript sources, provenance maps or large diagnostics and **does not** claim those commands pass on this public snapshot. Do not characterize an unavailable preflight as a failing previously certified experiment, and do not assume the omitted files are unavailable to the author or journal.

## 2. Independent clean-R1 PDF-to-JSON check: 28 public-core numeric cells

A new **standalone standard-library checker** exists at `revision_audits/audit_public_R1_core_table_gate.py`. It is outside frozen v70, reads only archived JSON, and requires neither TeX source nor SciPy/Torch or external datasets. It stores exactly the mean±population-SD entries printed in the assembled clean R1:

- **Table 1, assembled PDF page 78**: four objectives × four label budgets = **16 mean/SD cells**. All match `results/main_table_reconciliation_20261009/results.json` to the displayed single-decimal precision. This is the run that frozen metadata expressly identifies as superseding the earlier `results/main_alpha15` for Table 1.
- **Table 3, assembled PDF page 84**: three validation-selected robust loss families × four budgets = **12 mean/SD cells**. All match `results/validation_selected/validation_selection.json` for n=50/200/500 and `results/validation_selected_n100/validation_selection.json` for n=100. Each chosen variant is also checked as a maximizer of **validation**, not test, accuracy. Its five held-out-seed entries are reaggregated to check mean and population SD.

A separate independent live-GitHub JavaScript calculation checked **28/28** mean/SD entries against the clean PDF transcriptions and independently recomputed the seed-based population SD and validation argmax conditions. **The newly committed Python checker has not yet been executed against a complete local checkout**; no such execution is claimed. The checker can be run later with:

```sh
python revision_audits/audit_public_R1_core_table_gate.py
```

It intentionally prints a `SKIP` for the **full** Supplementary Materials audit when the unpublished inputs are missing, instead of silently asserting total coverage or overwriting sources.

## 3. What is verified versus unverified

**Verified from already-available source/data:** the 28 selected *clean manuscript* core numeric cells above; selected archived main and downstream provenance facts; separate prior audits have recomputed UCR, NOISEX, masking, Cauchy and normalization summaries at the **JSON level**.

**Not independently verified:** the exact current Supplementary Materials TeX/PDF transcription of every row in S1–S18, all figure captions, and absent score-sample SHA checks. The archival `audit_supplement_tables.py` describes substantial historical local checks, but the required complete sources are not in this public release and it cannot now be rerun against them. The assembled R1 PDF alone is **not** the supplementary source PDF.

**Reproducibility action, only if improving public release:** the new self-contained checker offers a narrowly scoped, transparent public numeric gate with no source-disclosure requirement. For a full supplementary table audit, explicitly provide the exact submitted supplementary artifact/TeX and its external table fragments in an authorized location or keep that checker marked as requiring private inputs. Do **not** modify the frozen v70 archive merely to make previously private files appear to have been shipped in 2026-10.

**Manuscript disposition:** no correction of the current Table 1 or Table 3 values is warranted; no new GPU experiment required. The useful distinction is **verified selected numeric evidence ≠ independently verified entire supplementary source package**. Any portal-submission status must be confirmed separately and is not inferable from a local PDF.


## 4. Editorial Manager assembled-R1 evidence (2026-11-14 addendum)

A separate read of the **94-page combined** `SIGPRO-D-26-02622_R1.pdf` establishes that this is not merely a local anonymous PDF: **assembled page 1** explicitly displays manuscript number `SIGPRO-D-26-02622R1` with an `Editorial Manager® and ProduXion Manager®` footer; **page 3** advertises an attached/linked supplementary item named **`Supplementary_Material_v73.pdf`**; page 4 is the revised cover letter, page 5 Highlights, and later pages contain marked/unmarked manuscripts. The cover letter states that the original alpha=1.5 results were withdrawn, the historical L1 inconsistency was documented, and the corrected `0.4–1.5 pp` gain is descriptive.

Thus, a journal-system **R1 assembly/preview artifact** and named supplementary upload slot are evidenced. This is not proof that the portal's **final submission step** was completed or that its supplementary bytes can be independently inspected. The assembled PDF's one-page filename reference does **not** reproduce the complete `Supplementary_Material_v73.pdf` pages. The named supplement could not be resolved as a separate readable Library/GitHub resource in the current audit. A subsequent programmatic materialization of the combined R1 PDF returned a temporary 429 upstream capacity error, so it did not establish whether an embedded-file object exists; **no absence claim** about embedded attachments should be made on that basis.

The proper distinction is:
- **Verified:** a journal-system R1 assembly exists and references a supplementary artifact by exact filename;
- **Unverified:** the supplementary file's contents, hashes, its final S1–S18 table/figure pagination, which response letter is attached, and the portal's completed-submission status.

Do not confuse `Supplementary_Material_v73.pdf` (a separately named editorial-system attachment) with the frozen public `revision_v70_20261010/` archive (a code/results release containing no manuscript TeX). No other file was silently substituted for the supplement. The 28/28 clean core-table checks remain valid with their stated limited scope.
