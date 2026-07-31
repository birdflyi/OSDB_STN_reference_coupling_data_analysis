"""Directed and undirected network-view helpers."""

from __future__ import annotations

import pandas as pd


def directed_to_undirected_edges(
    edges: pd.DataFrame,
    source_col: str = "source_repo",
    target_col: str = "target_repo",
    weight_col: str = "weight",
    drop_self_loop: bool = True,
) -> pd.DataFrame:
    """Derive an undirected edge table from direct directed reference edges."""

    missing = [col for col in (source_col, target_col, weight_col) if col not in edges.columns]
    if missing:
        raise KeyError(f"missing undirected-view columns: {', '.join(missing)}")

    work = edges.copy()
    if drop_self_loop:
        work = work[work[source_col].astype(str) != work[target_col].astype(str)]

    pairs = work[[source_col, target_col]].astype(str).apply(lambda row: sorted(row.tolist()), axis=1)
    work["node_u"] = pairs.map(lambda pair: pair[0])
    work["node_v"] = pairs.map(lambda pair: pair[1])
    work["_directed_edge_count"] = 1

    return (
        work.groupby(["node_u", "node_v"], dropna=False)
        .agg(weight=(weight_col, "sum"), directed_edge_count=("_directed_edge_count", "sum"))
        .reset_index()
        .sort_values(["node_u", "node_v"])
        .reset_index(drop=True)
    )
