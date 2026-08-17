"""RQ3 subdomain helpers."""

from __future__ import annotations

import re

import pandas as pd


MIXED_LABEL_PATTERN = re.compile(r"[;,|/]")


def filter_category_mode(
    records: pd.DataFrame,
    label_col: str = "category_label",
    mode: str = "include_mixed",
) -> pd.DataFrame:
    """Apply permissive or strict handling of mixed/multilabel categories."""

    if label_col not in records.columns:
        raise KeyError(f"missing category column: {label_col}")
    if mode == "include_mixed":
        return records.copy()
    if mode != "exclude_mixed_or_multilabel":
        raise ValueError("mode must be include_mixed or exclude_mixed_or_multilabel")

    labels = records[label_col].fillna("").astype(str)
    keep = ~labels.map(lambda value: bool(MIXED_LABEL_PATTERN.search(value)))
    return records[keep].copy()
