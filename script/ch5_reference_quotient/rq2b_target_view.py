"""RQ2b target-role summaries."""

from __future__ import annotations

import pandas as pd


def summarize_target_view(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize target in-degree and in-strength from a directed edge table."""

    required = {"source_repo", "target_repo", "weight"}
    missing = required - set(edges.columns)
    if missing:
        raise KeyError(f"missing target-view columns: {', '.join(sorted(missing))}")
    return (
        edges.groupby("target_repo")
        .agg(in_degree=("source_repo", "nunique"), in_strength=("weight", "sum"))
        .reset_index()
        .sort_values(["in_strength", "in_degree", "target_repo"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
