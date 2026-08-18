"""Freeze the seed-centered observation boundary for Reference Quotient."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from GH_CoRE.working_flow.query_OSDB_github_log import (
    get_repo_name_fileformat,
    get_repo_year_filename,
)


def repo_filename(repo_name: str, year: int) -> str:
    return get_repo_year_filename(get_repo_name_fileformat(str(repo_name)), year)


def relative_evidence_path(path: str | Path, source_root: str | Path) -> str:
    """Return a stable source-root-relative path for frozen seed outputs."""
    if not str(path):
        return ""
    return Path(path).resolve().relative_to(Path(source_root).resolve()).as_posix()


def build_seed_manifests(
    activity_path: str | Path,
    evidence_dir: str | Path,
    year: int,
    threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    activity = pd.read_csv(activity_path, dtype={"repo_id": "string"})
    activity["activity_count"] = pd.to_numeric(activity["i_pr_rec_cnt"], errors="coerce")
    candidates = activity[activity["repo_name"].notna() & (activity["activity_count"] >= threshold)].copy()
    candidates["repo_id"] = candidates["repo_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    candidates["evidence_filename"] = candidates["repo_name"].map(lambda value: repo_filename(str(value), year))

    directory = Path(evidence_dir)
    actual_by_lower = {path.name.lower(): path for path in directory.glob(f"*_{year}.csv")}
    candidates["evidence_available"] = candidates["evidence_filename"].str.lower().isin(actual_by_lower)
    candidates["evidence_path"] = candidates["evidence_filename"].str.lower().map(
        lambda name: str(actual_by_lower[name].resolve()) if name in actual_by_lower else ""
    )
    candidates["seed_boundary_reason"] = candidates["evidence_available"].map(
        {True: "activity_threshold_and_frozen_evidence_available", False: "candidate_missing_frozen_evidence"}
    )

    seeds = candidates[candidates["evidence_available"]].copy()
    seeds["seed_order"] = range(1, len(seeds) + 1)
    seed_columns = [
        "seed_order",
        "repo_id",
        "repo_name",
        "repo_name_used",
        "activity_count",
        "repo_created_at",
        "category_label",
        "evidence_filename",
        "evidence_path",
        "seed_boundary_reason",
    ]
    return seeds[seed_columns].reset_index(drop=True), candidates.reset_index(drop=True)


def assert_seed_boundary(seeds: pd.DataFrame, candidates: pd.DataFrame, expected_seeds: int, expected_candidates: int) -> None:
    if len(candidates) != expected_candidates:
        raise ValueError(f"candidate seed count drift: expected {expected_candidates}, got {len(candidates)}")
    if len(seeds) != expected_seeds:
        raise ValueError(f"analysis seed count drift: expected {expected_seeds}, got {len(seeds)}")
    if seeds["repo_id"].duplicated().any():
        raise ValueError("analysis seed repo_id must be unique")
    if seeds["repo_name"].str.lower().duplicated().any():
        raise ValueError("analysis seed repo_name must be unique")
