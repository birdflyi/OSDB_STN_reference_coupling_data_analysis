"""Project-level direct edge table construction."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .seed_selection import normalize_repo_identity


def build_direct_edge_table(
    records: pd.DataFrame,
    source_col: str = "source_repo",
    target_col: str = "target_repo",
    analysis_seed_repos: Optional[Iterable[object]] = None,
    drop_self_loop: bool = False,
    drop_unresolved_target: bool = True,
) -> pd.DataFrame:
    """Aggregate direct Reference records into directed weighted repo edges."""

    missing = [col for col in (source_col, target_col) if col not in records.columns]
    if missing:
        raise KeyError(f"missing edge columns: {', '.join(missing)}")

    work = records.copy()
    if drop_unresolved_target:
        keep = work[target_col].notna() & (work[target_col].astype(str).str.strip() != "")
        work = work[keep]

    work["_source_repo"] = work[source_col].astype(str)
    work["_target_repo"] = work[target_col].astype(str)
    work["_record_weight"] = 1

    grouped = (
        work.groupby(["_source_repo", "_target_repo"], dropna=False)
        .agg(weight=("_record_weight", "sum"), multiplicity=("_record_weight", "size"))
        .reset_index()
        .rename(columns={"_source_repo": "source_repo", "_target_repo": "target_repo"})
    )
    grouped["is_self_loop"] = grouped["source_repo"] == grouped["target_repo"]
    if drop_self_loop:
        grouped = grouped[~grouped["is_self_loop"]].copy()

    seed_set = {normalize_repo_identity(v) for v in analysis_seed_repos or []}
    grouped["source_is_seed"] = grouped["source_repo"].map(lambda value: normalize_repo_identity(value) in seed_set)
    grouped["target_is_seed"] = grouped["target_repo"].map(lambda value: normalize_repo_identity(value) in seed_set)

    return grouped[
        [
            "source_repo",
            "target_repo",
            "weight",
            "multiplicity",
            "is_self_loop",
            "source_is_seed",
            "target_is_seed",
        ]
    ].reset_index(drop=True)
