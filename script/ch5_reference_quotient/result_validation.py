"""Validation helpers for preparation-stage outputs."""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd


EDGE_COLUMNS = {
    "source_repo",
    "target_repo",
    "weight",
    "multiplicity",
    "is_self_loop",
    "source_is_seed",
    "target_is_seed",
}


def validate_edge_table(edges: pd.DataFrame) -> List[str]:
    """Return validation errors for the direct edge table."""

    errors: List[str] = []
    missing = EDGE_COLUMNS - set(edges.columns)
    if missing:
        errors.append(f"missing edge columns: {', '.join(sorted(missing))}")
    if "weight" in edges.columns and (edges["weight"] <= 0).any():
        errors.append("edge weights must be positive")
    if "multiplicity" in edges.columns and (edges["multiplicity"] <= 0).any():
        errors.append("edge multiplicity must be positive")
    return errors


def assert_not_eligible_for_freeze(eligible_for_freeze: bool) -> None:
    """Protect the preparation stage from accidentally freezing outputs."""

    if eligible_for_freeze:
        raise AssertionError("preparation-stage outputs must not be eligible for freeze")
