"""Weighted directed Reference Quotient edge construction."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Optional

import pandas as pd
from script.complex_network_analysis.build_network.build_Graph import build_Graph

from .membership import unique_project_membership


def build_reference_quotient_edges(
    records: pd.DataFrame,
    source_membership_col: str = "src_entity_id_agg",
    target_membership_col: str = "tar_entity_id_agg",
    analysis_seed_ids: Optional[Iterable[object]] = None,
    preserve_self_loops: bool = True,
) -> pd.DataFrame:
    """Compute Q=M^T R_P M for a DataFrame of fine-grained records."""

    missing = [col for col in (source_membership_col, target_membership_col) if col not in records]
    if missing:
        raise KeyError(f"missing membership columns: {', '.join(missing)}")
    work = records.copy()
    work["source_project_id"] = work[source_membership_col].map(unique_project_membership)
    work["target_project_id"] = work[target_membership_col].map(unique_project_membership)
    work = work[work["source_project_id"].notna() & work["target_project_id"].notna()]
    counter = Counter(zip(work["source_project_id"], work["target_project_id"]))
    return edge_frame(counter, analysis_seed_ids or (), preserve_self_loops=preserve_self_loops)


def edge_frame(
    counter: Mapping[tuple[str, str], int],
    analysis_seed_ids: Iterable[object],
    preserve_self_loops: bool = True,
) -> pd.DataFrame:
    seed_set = {str(value) for value in analysis_seed_ids}
    rows = []
    for (source, target), weight in sorted(counter.items()):
        self_loop = source == target
        if self_loop and not preserve_self_loops:
            continue
        rows.append(
            {
                "source_project_id": source,
                "target_project_id": target,
                "weight": int(weight),
                "multiplicity": int(weight),
                "is_self_loop": self_loop,
                "source_is_seed": source in seed_set,
                "target_is_seed": target in seed_set,
            }
        )
    return pd.DataFrame(rows)


def build_direct_edge_table(
    records: pd.DataFrame,
    source_col: str = "source_repo",
    target_col: str = "target_repo",
    analysis_seed_repos: Optional[Iterable[object]] = None,
    drop_self_loop: bool = False,
    drop_unresolved_target: bool = True,
) -> pd.DataFrame:
    """Compatibility helper for callers that already have project columns."""

    missing = [col for col in (source_col, target_col) if col not in records]
    if missing:
        raise KeyError(f"missing edge columns: {', '.join(missing)}")
    work = records.copy()
    if drop_unresolved_target:
        target = work[target_col]
        work = work[target.notna() & target.astype(str).str.strip().ne("")]
    work[source_col] = work[source_col].astype(str)
    work[target_col] = work[target_col].astype(str)
    if work.empty:
        counter = {}
    else:
        graph = build_Graph(
            work[[source_col, target_col]],
            src_tar_colnames=[source_col, target_col],
            default_node_types=[None, None],
            default_edge_type=None,
            init_edge_weight=True,
            w_trunc=1,
            out_g_type="DG",
        )
        counter = {
            (str(source), str(target)): int(data["weight"])
            for source, target, data in graph.edges(data=True)
        }
    result = edge_frame(counter, analysis_seed_repos or (), preserve_self_loops=not drop_self_loop)
    return result.rename(
        columns={"source_project_id": "source_repo", "target_project_id": "target_repo"}
    )
