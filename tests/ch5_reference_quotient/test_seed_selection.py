from pathlib import Path

from script.ch5_reference_quotient.config import load_config, resolved_inputs
from script.ch5_reference_quotient.seed_selection import (
    assert_seed_boundary,
    build_seed_manifests,
    repo_filename,
)


def test_repo_filename_uses_gh_core_convention():
    assert repo_filename("owner/repo", 2023) == "owner_repo_2023.csv"
    assert repo_filename("group/subgroup/repo", 2024) == "group_subgroup_repo_2024.csv"


def test_frozen_seed_boundary_is_301_candidates_and_294_observed_sources():
    config = load_config("configs/ch5_reference_quotient_p0.yaml")
    inputs = resolved_inputs(config)
    seeds, candidates = build_seed_manifests(
        inputs["repo_activity_statistics"],
        inputs["gh_core_ref_node_agg_dir"],
        config.get_int("study_year"),
        config.get_int("analysis_seed_activity_threshold"),
    )
    assert_seed_boundary(seeds, candidates, 294, 301)
    assert len(seeds) == 294
    assert (candidates.evidence_available == False).sum() == 7  # noqa: E712
    assert all(Path(path).is_file() for path in seeds.evidence_path)
