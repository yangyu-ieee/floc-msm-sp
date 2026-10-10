# Revised SP manuscript: code and archived results

This versioned directory accompanies the v70 revised-manuscript candidate dated 2026-10-10. It provides corrected experiment code and selected archived result records for inspection and targeted auditing. It is not a statement that all GPU experiments were independently rerun, and the archive is not an author-approved or Editorial Manager-verified submission.

## Important version boundary

The repository root contains the original, superseded submission code. In particular, do not use the root-level `floc_msm_main.py` or `floc_10seed.py` to reproduce the revised alpha=1.5 results. The affected original alpha=1.5 results and significance claims were withdrawn after an error was found in the Chambers–Mallows–Stuck sampler. The corrected sampler is in `code/stable_noise.py`; its tests are in `code/tests/`.

The revised primary benchmark is defined by `code/floc_msm_main_corrected.py` and `code/run_main_table_reconciliation.py`, with the archived run under `results/main_table_reconciliation_20261009/`. It comprises five downstream seeds conditional on one pretrained checkpoint per loss; do not interpret the downstream seeds as independent pretraining replications.

## Contents

- `code/`: corrected benchmark runners, recovered auxiliary runners, audit scripts, and sampler tests.
- `results/`: selected JSON/TXT/Markdown run records and summaries from the v70 evidence archive. Raw datasets, manuscript/rebuttal files, machine-specific logs, and large intermediate diagnostic arrays are not included.
- `requirements*.txt`: recorded CUDA-oriented dependencies, a CPU alternative, and additional UCR dependencies.
- `MANIFEST_SHA256.txt`: deterministic SHA-256 listing for files in this versioned directory, excluding the manifest itself.

## Environment and checks

Use Python 3.10 or newer. Install the environment appropriate to your machine:

```bash
python -m pip install -r requirements-cpu.txt
```

For a CUDA environment, see `requirements.txt`; for UCR experiments, install `requirements-ucr.txt` as well. CUDA/PyTorch versions may need to match the local driver.

From this directory, run the data-free checks:

```bash
python -m unittest discover -s code/tests -v
python code/run_smoke_tests.py
```

The following archived-record checks were run against this public snapshot:

```bash
python code/audit_main_table_summary.py
python code/audit_alpha1_training_dynamics.py
python code/audit_cauchy_unpaired_welch.py
python code/make_manifest.py --check
```

Other audit scripts from the author-review archive are included for reference, but some require private manuscript snapshots, a local source-provenance file, or large intermediate diagnostic arrays that are intentionally not redistributed here. Those commands are not claimed to pass in this public snapshot. None of the checks above retrains the complete GPU experiments.

## Protocol and interpretation notes

- The primary synthetic comparison uses fixed-count masking (16 of 32 tokens). Stagewise diagnostics and the Huber/Charbonnier/MCC validation-selected sensitivity runs use Bernoulli(0.5) masking. The latter are separate runs and are not matched replacements for the primary benchmark.
- UCR and NOISEX-UCR use dataset-specific sequence lengths and Bernoulli token masking; CWRU uses its separate length-1024 stress-test protocol. See the corresponding runner and archived configuration files for details.
- The UCR summary is the equal-weight mean of ordinary test accuracy over 20 dataset–shot conditions; it is neither class-macro accuracy nor pooled test accuracy.
- The archived validation sensitivity records include candidate validation results and selected settings. Test results must not be used to choose hyperparameters.
- External datasets are not redistributed. Obtain them from their original sources and follow their access and licensing terms.
- The historical exact-count random-token masking runner was not recovered. Its archived configuration and output records are not represented as a fully rerunnable source release.
- The public copy of the alpha=1 base runner replaces its machine-specific default output directory with a path relative to the release. Its original run-source SHA-256 remains in the archived configuration; a separate hash identifies the public portability-adjusted copy. No experiment logic was changed in that path-only adjustment.
- Names such as `floc_1.2` may remain in historical serialized records as provenance identifiers. The manuscript terminology is “FLOM-based p-moment loss”; FLOC is reserved for fractional lower-order covariance.

## Citation and scope

Please cite the associated revised manuscript if you use this release. This code package is intended to make the revised experiments and their limitations easier to inspect; it does not supersede the original historical files or imply endorsement of any unverified rerun.
