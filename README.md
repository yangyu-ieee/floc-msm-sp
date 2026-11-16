# FLOC-MSM: Robust Masked Signal Modeling under Impulsive Noise

**Manuscript:** “Robust Masked Signal Modeling under Impulsive Noise: A Fractional Lower-Order Perspective”  
**Journal:** *Signal Processing* (Elsevier)  
**Manuscript ID:** SIGPRO-D-26-02622

## Start here

- [R1 revision status and evidence map](REVISION_STATUS.md) — current verified/unverified status, claim boundaries, missing source inputs and repository maintenance findings.
- [R1 audit index](revision_audits/README.md) — navigation across the 18 topic-specific audit records.
- [Corrected v70 code/results archive](revision_v70_20261010/README.md) — versioned corrected sampler, experiment runners, selected archived outputs, tests and manifest.

## Repository version boundaries

There are three distinct layers:

1. **Root-level scripts:** historical pre-revision submission code. The affected original alpha=1.5 CMS sampler used the wrong angular factor, cos(U) instead of cos((1-alpha)U). The affected original alpha=1.5 results and associated significance claims were withdrawn. These files are retained for provenance only.
2. **revision_v70_20261010/:** frozen corrected code and selected archived results. This is a public reproducibility snapshot, not the full manuscript/rebuttal/supplement package and not proof that every GPU experiment was independently rerun.
3. **revision_audits/:** post-assembly source, protocol, statistical and editorial audits. These notes do not silently amend the frozen v70 data or the assembled R1 manuscript.

Do not use the root-level scripts to reproduce the revised results. The old scripts include machine-specific paths and/or superseded experimental logic; they are not a portable clean-checkout entrypoint. In particular, the former README commands prefixed them with a non-existent code/ directory. No legacy Quick Start command is provided here.

## Reproducing the corrected public snapshot

Follow the environment, test and audit commands in [revision_v70_20261010/README.md](revision_v70_20261010/README.md). Run those commands from inside revision_v70_20261010/, not from the repository root.

The narrow public-core Table 1/Table 3 checker is [revision_audits/audit_public_R1_core_table_gate.py](revision_audits/audit_public_R1_core_table_gate.py). It checks selected clean-R1 numeric cells only; it is not a complete paper or Supplementary Materials audit.

## What is not included in this Git repository

The public tree does not contain the complete clean R1 manuscript source/PDF, the separately named Supplementary_Material_v73.pdf, all supplementary TeX/table fragments, the full source-provenance map, or all large intermediate diagnostic arrays. The 94-page Editorial Manager assembly and the exact final portal state are not established by the code archive alone. See [REVISION_STATUS.md](REVISION_STATUS.md) before making any claim about submission readiness.

## Data and licensing

- UCR datasets are loaded through aeon; obtain them from their original source and follow their terms.
- CWRU bearing data and NOISEX-92 are not redistributed in this repository.
- Synthetic experiments generate their signals/noise through the archived code.

The repository is licensed under the MIT License; see [LICENSE](LICENSE).

## Citation

If using the corrected code/results snapshot, cite the associated revised manuscript and identify the exact archive version and experiment protocol. Do not cite the root-level historical scripts as the source of the revised results.
