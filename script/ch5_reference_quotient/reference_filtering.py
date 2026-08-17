"""Direct Reference filtering and small deduplication helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def filter_direct_references(
    records: pd.DataFrame,
    relation_col: str = "relation_type",
    relation_type: str = "Reference",
) -> pd.DataFrame:
    """Keep only direct Reference records."""

    if relation_col not in records.columns:
        raise KeyError(f"missing relation column: {relation_col}")
    return records[records[relation_col] == relation_type].copy()


def drop_unresolved_targets(records: pd.DataFrame, target_col: str = "target_repo") -> pd.DataFrame:
    """Drop records whose target repo cannot be resolved."""

    if target_col not in records.columns:
        raise KeyError(f"missing target column: {target_col}")
    target = records[target_col]
    keep = target.notna() & (target.astype(str).str.strip() != "")
    return records[keep].copy()


def deduplicate_references(
    records: pd.DataFrame,
    rule: str = "event_source_target",
    subset_cols: Sequence[str] = ("event_id", "source_repo", "target_repo"),
) -> pd.DataFrame:
    """Deduplicate reference records under an explicit preparation rule."""

    if rule == "none":
        return records.copy()
    if rule != "event_source_target":
        raise ValueError("rule must be 'none' or 'event_source_target'")
    missing = [col for col in subset_cols if col not in records.columns]
    if missing:
        raise KeyError(f"missing dedup columns: {', '.join(missing)}")
    return records.drop_duplicates(subset=list(subset_cols), keep="first").copy()


def flag_self_loops(
    records: pd.DataFrame,
    source_col: str = "source_repo",
    target_col: str = "target_repo",
) -> pd.DataFrame:
    """Add an is_self_loop flag based on source and target repo identity."""

    if source_col not in records.columns or target_col not in records.columns:
        raise KeyError("missing source or target column")
    result = records.copy()
    result["is_self_loop"] = result[source_col].astype(str) == result[target_col].astype(str)
    return result


def filter_external_service_targets(
    records: pd.DataFrame,
    target_type_col: str = "target_type",
    excluded_types: Iterable[str] = (),
) -> pd.DataFrame:
    """Remove records whose target type belongs to external service placeholders."""

    if target_type_col not in records.columns:
        return records.copy()
    excluded = set(excluded_types)
    return records[~records[target_type_col].isin(excluded)].copy()
