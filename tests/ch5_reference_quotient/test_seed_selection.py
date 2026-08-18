from pathlib import Path

from script.ch5_reference_quotient.config import load_config, resolved_inputs
from script.ch5_reference_quotient.seed_selection import (
    assert_seed_boundary,
    build_seed_manifests,
    repo_filename,
    relative_evidence_path,
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


def test_seed_evidence_paths_have_stable_source_relative_serialization():
    config = load_config("configs/ch5_reference_quotient_p0.yaml")
    inputs = resolved_inputs(config)
    seeds, candidates = build_seed_manifests(
        inputs["repo_activity_statistics"],
        inputs["gh_core_ref_node_agg_dir"],
        config.get_int("study_year"),
        config.get_int("analysis_seed_activity_threshold"),
    )
    source_root = config.source_repository["path"]
    for frame in (seeds, candidates):
        serialized = frame["evidence_path"].map(lambda path: relative_evidence_path(path, source_root))
        assert all(not Path(path).is_absolute() for path in serialized if path)
        assert all("D:/" not in path and "C:/" not in path for path in serialized if path)
        assert all(path.startswith("data/") for path in serialized if path)


def test_seed_audit_serialization_is_deterministic(tmp_path):
    config = load_config("configs/ch5_reference_quotient_p0.yaml")
    inputs = resolved_inputs(config)
    seeds, candidates = build_seed_manifests(
        inputs["repo_activity_statistics"],
        inputs["gh_core_ref_node_agg_dir"],
        config.get_int("study_year"),
        config.get_int("analysis_seed_activity_threshold"),
    )
    source_root = config.source_repository["path"]
    for name, frame in (
        ("analysis_seed_manifest_294.csv", seeds),
        ("candidate_seed_observation_audit.csv", candidates),
    ):
        serialized = frame.copy()
        serialized["evidence_path"] = serialized["evidence_path"].map(
            lambda path: relative_evidence_path(path, source_root)
        )
        first = tmp_path / f"first_{name}"
        second = tmp_path / f"second_{name}"
        serialized.to_csv(first, index=False)
        serialized.to_csv(second, index=False)
        assert first.read_bytes() == second.read_bytes()
