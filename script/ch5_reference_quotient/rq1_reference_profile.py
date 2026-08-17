"""Small RQ1 reference-profile summaries."""

from __future__ import annotations

import pandas as pd


def value_distribution(records: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return count and share for a categorical column."""

    if column not in records.columns:
        raise KeyError(f"missing column: {column}")
    counts = records[column].fillna("UNKNOWN").value_counts(dropna=False).rename_axis(column).reset_index(name="count")
    total = counts["count"].sum()
    counts["share"] = counts["count"] / total if total else 0
    return counts
