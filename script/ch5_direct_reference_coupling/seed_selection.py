"""Seed repository selection helpers."""

from __future__ import annotations

from typing import Iterable, Set

import pandas as pd


def select_analysis_seed_repos(
    repos: pd.DataFrame,
    threshold: int = 10,
    activity_col: str = "i_pr_rec_cnt",
) -> pd.DataFrame:
    """Return repos whose Issue/Pull Request activity count reaches the threshold."""

    if activity_col not in repos.columns:
        raise KeyError(f"missing activity column: {activity_col}")
    selected = repos[repos[activity_col] >= threshold].copy()
    if "repo_id" in selected.columns:
        selected["repo_id"] = selected["repo_id"].astype(str)
    return selected


def split_candidate_and_analysis_seeds(
    repos: pd.DataFrame,
    threshold: int = 10,
    activity_col: str = "i_pr_rec_cnt",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the candidate set and the activity-filtered analysis seed set."""

    candidate = repos.copy()
    analysis = select_analysis_seed_repos(candidate, threshold=threshold, activity_col=activity_col)
    return candidate, analysis


def normalize_repo_identity(value: object) -> str:
    """Normalize repo names or ids for stable set membership checks."""

    return str(value).strip().lower()


def mark_seed_roles(
    records: pd.DataFrame,
    analysis_seed_repos: Iterable[object],
    source_col: str = "source_repo",
    target_col: str = "target_repo",
) -> pd.DataFrame:
    """Annotate whether source and target repos are analysis seeds."""

    seed_set: Set[str] = {normalize_repo_identity(v) for v in analysis_seed_repos}
    result = records.copy()
    result["source_is_seed"] = result[source_col].map(lambda value: normalize_repo_identity(value) in seed_set)
    result["target_is_seed"] = result[target_col].map(lambda value: normalize_repo_identity(value) in seed_set)
    return result
