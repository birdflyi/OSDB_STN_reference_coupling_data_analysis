"""Repo-level reference aggregation helpers."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def coalesce_repo_identity(
    records: pd.DataFrame,
    output_col: str,
    candidate_cols: Sequence[str],
) -> pd.DataFrame:
    """Coalesce several repo id/name columns into a single repo identity column."""

    missing = [col for col in candidate_cols if col not in records.columns]
    if missing:
        raise KeyError(f"missing candidate columns: {', '.join(missing)}")
    result = records.copy()
    value = pd.Series([pd.NA] * len(result), index=result.index, dtype="object")
    for col in candidate_cols:
        value = value.fillna(result[col])
    result[output_col] = value
    return result


def ensure_repo_columns(
    records: pd.DataFrame,
    source_col: str = "source_repo",
    target_col: str = "target_repo",
) -> pd.DataFrame:
    """Validate and normalize source/target repo columns."""

    missing = [col for col in (source_col, target_col) if col not in records.columns]
    if missing:
        raise KeyError(f"missing repo columns: {', '.join(missing)}")
    result = records.copy()
    result[source_col] = result[source_col].astype("string")
    result[target_col] = result[target_col].astype("string")
    return result
