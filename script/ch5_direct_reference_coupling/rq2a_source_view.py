"""RQ2a source-role summaries."""

from __future__ import annotations

import pandas as pd


def summarize_source_view(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize source out-degree and out-strength from a directed edge table."""

    required = {"source_repo", "target_repo", "weight"}
    missing = required - set(edges.columns)
    if missing:
        raise KeyError(f"missing source-view columns: {', '.join(sorted(missing))}")
    return (
        edges.groupby("source_repo")
        .agg(out_degree=("target_repo", "nunique"), out_strength=("weight", "sum"))
        .reset_index()
        .sort_values(["out_strength", "out_degree", "source_repo"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
