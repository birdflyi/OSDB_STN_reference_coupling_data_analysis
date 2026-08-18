# OSDB_RefQ

OSDB_RefQ provides the Reference Quotient analysis and reproducible P0 pipeline for studying project-level explicit Reference relations in the open-source DBMS ecosystem.

**Reference Quotient (RefQ) is the project's current principal construct.** The analysis follows this evidence-preserving chain:

```text
Fine-grained Reference evidence
-> artifact-to-project membership
-> membership-induced graph coarsening
-> Project-level Reference Quotient Network (RefQN)
```

Reference Coupling is retained as related-work terminology and historical project context; the current analysis is formalized using Reference Quotient.

## Method and scope

The project-level quotient is defined as:

```text
Q = M^T R_P M
```

This construction produces a weighted, directed Project-level RefQN while preserving:

- Reference direction;
- Reference multiplicity and edge weight;
- artifact-to-project membership;
- provenance traceability back to fine-grained evidence.

The current primary experiment does not construct or analyze `QQ^T`, `Q^T Q`, shared-reference projections, bibliographic coupling, co-citation, composite centrality, or other network projections.

### Observation boundary

The Chapter 5 P0 network is a **seed-centered observed RefQN**, not a complete GitHub ecosystem network.

- Study year: 2023
- Candidate seeds: 301
- Analysis seeds: 294

Target projects may enter the expanded network because they are referenced by seed projects. Their presence does not imply that their source behavior was completely observed.

## Reproducibility

The upstream [open-source DBMS project list](https://github.com/birdflyi/od_label_issue_gen/tree/main/data/database_repo_label_dataframe) is updated monthly and may continue to evolve. It is separate from the frozen inputs used for the Chapter 5 P0 results.

The P0 results are reproduced from the frozen 2023 seed manifest, checksummed inputs, analytical configuration, and implementation provenance recorded by the generated P0 `manifest.json`. The frozen settings and source snapshot are declared in [`configs/ch5_reference_quotient_p0.yaml`](configs/ch5_reference_quotient_p0.yaml).

Provenance is divided between:

- Current implementation: `birdflyi/OSDB_RefQ`, branch `main`.
- Frozen input/source snapshot: the repository path, branch, and commit recorded in the P0 configuration. Its historical branch name is provenance and does not describe the current theoretical construct.

## Quick start

Run the RefQ test suite:

```bash
python -m pytest -p no:cacheprovider -q tests/ch5_reference_quotient
```

Run the P0 pipeline:

```bash
python -m script.ch5_reference_quotient.cli \
  --config configs/ch5_reference_quotient_p0.yaml \
  --execute
```

A repository clone alone is not sufficient to reproduce the frozen P0 outputs. Before execution, obtain and stage the required input data and frozen source snapshot at the locations recorded in the configuration. The run writes checksums and implementation provenance to its output manifest.

## Data

Campus intranet or logged-in VPN:

```bash
curl --output-dir "/path/to/your/Downloads_folder" -O http://49.52.27.60:8000/github_osdb_data-GH_CoRE-2.3.0-Pub-Alpha.tar.gz
```

Public access: [Google Drive](https://drive.google.com/file/d/1gBUOnCO1FmEbzF018BOt2H3Y1VIn5MbR/view?usp=sharing), [Zenodo](https://doi.org/10.5281/zenodo.18817348).
