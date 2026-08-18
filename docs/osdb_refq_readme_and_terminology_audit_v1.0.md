# OSDB_RefQ README and Terminology Audit v1.0

## Result

`README_UPDATE=PASS`

The public repository description now presents Reference Quotient (RefQ) as the current principal construct. No analysis logic, P0 values, data, tests, or configuration semantics were changed.

One non-blocking provenance item requires human review: `data/github_osdb_data/README.txt` records `OSDB_STN_RC_DA-0.1.0` as a release tag, but that tag is absent from both local refs and `origin` as checked with `git tag --list` and `git ls-remote --tags origin`.

## Baseline and scope

- Repository: `birdflyi/OSDB_RefQ`
- Branch: `main`
- Current HEAD: `27b2414a2e0ffb5f9994f561c9faaad944d39ac8`
- Initial worktree: clean and synchronized with `origin/main`
- Files changed by this task: `README.md` and this audit report
- Protected in this task: RefQ implementation, tests, P0 configuration semantics, outputs, data, manuscript, and dissertation
- Push status: not pushed

## README update summary

### Old to new public positioning

| Area | Old positioning | New positioning |
| --- | --- | --- |
| Repository purpose | A reference coupling data analysis project | Reference Quotient analysis and a reproducible Chapter 5 P0 pipeline |
| Main construct | Reference coupling | Reference Quotient (RefQ) and the Project-level RefQN |
| Construction | Not stated | Fine-grained evidence -> membership -> graph coarsening -> Project-level RefQN |
| Formal object | Not stated | `Q = M^T R_P M`, a weighted directed quotient network |
| Method boundary | Not stated | Explicitly excludes `QQ^T`, `Q^T Q`, shared-reference projection, bibliographic coupling, co-citation, composite centrality, and other projections |
| Observation boundary | Not stated | Seed-centered observed 2023 network with 301 candidate and 294 analysis seeds |
| Reproducibility | Dynamic project list only | Dynamic upstream list separated from frozen P0 config, inputs, checksums, and manifest provenance |
| Execution | No test or pipeline command | RefQ test command, P0 CLI command, and input-preparation warning |

The existing campus/VPN URL, Google Drive URL, Zenodo DOI, and dynamic upstream project-list URL were retained unchanged.

## Audit method

The audit used Git-native searches (`git grep -n -I`) over tracked text files. It did not rely on `rg`. Searches were case-insensitive where appropriate and covered:

- `OSDB_STN_reference_coupling_data_analysis`
- `direct_reference_coupling`
- `Direct Reference Coupling`
- `reference coupling`
- `引用耦合`
- `network projection`
- `Reference openness`
- `time evolution`
- `knowledge absorption`
- `knowledge cohesion`

A supplementary Git search covered hyphenated historical names such as `ch5-direct-reference-coupling-prep`, and the data metadata was checked for `OSDB_STN_RC_DA-0.1.0`.

Excluded as instructed:

- `.git/` and binary files;
- generated outputs under `outputs/` and data result directories;
- generated runtime records under `logs/`;
- this audit report itself, because it quotes the audited terms and would make the count self-referential.

`data/github_osdb_data/README.txt` was reviewed separately because it is text provenance metadata rather than a binary or generated analytical result.

## Residual classification

Counts below are final residual file-line hits after the README update. A line matching more than one search expression is counted once.

| Category | Count |
| --- | ---: |
| `MUST_UPDATE` | 0 |
| `ALLOWED_HISTORICAL` | 8 |
| `ALLOWED_RELATED_WORK` | 2 |
| `NEGATIVE_TEST` | 1 |
| `REQUIRES_REVIEW` | 1 |
| **Total** | **12** |

The original stale positioning at the former `README.md:2` was the sole `MUST_UPDATE` finding and has been resolved.

### Per-hit disposition

| Category | Location | Term/context | Disposition |
| --- | --- | --- | --- |
| `ALLOWED_HISTORICAL` | `configs/ch5_reference_quotient_p0.yaml:25` | `ch5-direct-reference-coupling-prep` | Frozen source snapshot branch; retain exactly as provenance. |
| `ALLOWED_HISTORICAL` | `reference_descriptive_analysis.py:4559` | `Time Evolution` | Legacy descriptive workflow outside `script/ch5_reference_quotient`; no P0 import or semantic effect. |
| `ALLOWED_HISTORICAL` | `reference_descriptive_analysis.py:4565` | `Time Evolution` | Same legacy visualization context. |
| `ALLOWED_HISTORICAL` | `reference_descriptive_analysis.py:4568` | `Time Evolution` | Same legacy visualization context. |
| `ALLOWED_HISTORICAL` | `reference_descriptive_analysis.py:5749` | `引用耦合` | Legacy root analysis workflow text, outside the RefQ P0 package. |
| `ALLOWED_HISTORICAL` | `reference_descriptive_analysis.py:5871` | `引用耦合` | Legacy descriptive log text, outside the RefQ P0 package. |
| `ALLOWED_HISTORICAL` | `reference_descriptive_analysis.py:5908` | `引用耦合` | Legacy topology log text, outside the RefQ P0 package. |
| `ALLOWED_HISTORICAL` | `reference_descriptive_analysis.py:5910` | `引用耦合` | Legacy topology log text, outside the RefQ P0 package. |
| `ALLOWED_RELATED_WORK` | `README.md:14` | `Reference Coupling` | Explicitly and correctly limited to related-work terminology and historical context. |
| `ALLOWED_RELATED_WORK` | `README.md:31` | `network projections` | Contrastive method-boundary statement saying these projections are not part of the primary experiment. |
| `NEGATIVE_TEST` | `tests/ch5_reference_quotient/test_config_manifest_cli.py:26` | `direct_reference_coupling` | Negative assertion preventing the retired CLI name from returning. |
| `REQUIRES_REVIEW` | `data/github_osdb_data/README.txt:3` | `OSDB_STN_RC_DA-0.1.0` | Metadata calls this a release tag, but no matching local or `origin` tag was found. Preserve until historical provenance is confirmed. |

The tracked generated `logs/Run_Info.log` also contains old repository paths and historical terminology. It was excluded from residual counts as a generated runtime record and does not define the current public or P0 contract.

## Code and configuration assessment

No substantive RefQ code or configuration issue was found.

- The RefQ P0 implementation contains no positive use of the retired `direct_reference_coupling` name.
- The one test occurrence is a negative guard.
- The old branch name in the P0 configuration is required frozen source-data provenance and must not be rewritten.
- The root legacy descriptive workflow contains historical terminology but is not imported by the RefQ P0 package or its tests.
- The unverified legacy release-tag identifier is a documentation/provenance review item, not a P0 computational issue.

No code or configuration change is recommended in this documentation-only task.

## README diff summary

`git diff --stat -- README.md`:

```text
README.md | 81 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++----
1 file changed, 76 insertions(+), 5 deletions(-)
```

The diff replaces the stale one-line project description, preserves all existing download destinations, and adds concise sections for method scope, observation boundary, reproducibility, provenance, quick start, and data access.

## Final worktree state

Expected documentation-only state after generation:

```text
 M README.md
?? docs/osdb_refq_readme_and_terminology_audit_v1.0.md
```

No local commit was created. Human review is recommended before committing and pushing, with particular attention to the unverified legacy release-tag identifier.
