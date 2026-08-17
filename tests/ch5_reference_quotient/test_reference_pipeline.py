import pandas as pd

from script.ch5_reference_quotient.edge_table import (
    build_direct_edge_table,
    build_reference_quotient_edges,
)
from script.ch5_reference_quotient.membership import (
    MembershipRegistry,
    canonical_project_entity_identity,
    classify_membership,
    normalized_entity_identity,
    unique_project_membership,
)
from script.ch5_reference_quotient.statistics import kruskal_fdr


def test_direct_edge_table_uses_shared_graph_aggregation():
    records = pd.DataFrame(
        [
            {"source": "a", "target": "b"},
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"},
            {"source": "a", "target": "a"},
        ]
    )

    edges = build_direct_edge_table(records, source_col="source", target_col="target")

    assert edges[["source_repo", "target_repo", "weight"]].to_dict("records") == [
        {"source_repo": "a", "target_repo": "a", "weight": 1},
        {"source_repo": "a", "target_repo": "b", "weight": 2},
        {"source_repo": "b", "target_repo": "a", "weight": 1},
    ]


def test_unique_membership_and_q_equals_mt_rp_m_aggregation():
    records = pd.DataFrame(
        [
            {"src_entity_id_agg": "R_1", "tar_entity_id_agg": "R_2"},
            {"src_entity_id_agg": "R_1", "tar_entity_id_agg": "R_2"},
            {"src_entity_id_agg": "R_2", "tar_entity_id_agg": "R_1"},
            {"src_entity_id_agg": "R_1", "tar_entity_id_agg": "R_1"},
            {"src_entity_id_agg": "R_1", "tar_entity_id_agg": "Object"},
        ]
    )
    edges = build_reference_quotient_edges(records, analysis_seed_ids=["1"], preserve_self_loops=True)
    edge_12 = edges[(edges.source_project_id == "1") & (edges.target_project_id == "2")].iloc[0]
    edge_21 = edges[(edges.source_project_id == "2") & (edges.target_project_id == "1")].iloc[0]
    loop = edges[(edges.source_project_id == "1") & (edges.target_project_id == "1")].iloc[0]
    assert edge_12.weight == 2
    assert edge_21.weight == 1
    assert loop.weight == 1 and bool(loop.is_self_loop)
    assert len(edges) == 3


def test_membership_parser_rejects_unresolved_and_ambiguous_values():
    assert unique_project_membership("R_42") == "42"
    assert unique_project_membership("R_1 and R_2") is None
    assert classify_membership("Object", "Object") == "non_project"
    assert classify_membership("", "Repo") == "unresolved"
    assert classify_membership("R_1 and R_2", "Repo") == "ambiguous"
    assert normalized_entity_identity("PRR_None") is None
    assert normalized_entity_identity("PRRC_None") is None
    assert normalized_entity_identity("I_42#7") == "I_42#7"
    assert canonical_project_entity_identity(None, "R_42") == "R_42"
    assert canonical_project_entity_identity("PRR_None", "R_42") == "R_42"
    assert canonical_project_entity_identity(None, "R_1 and R_2") is None


def test_membership_registry_reports_conflicts_for_exclusion(tmp_path):
    registry = MembershipRegistry(tmp_path / "memberships.sqlite")
    registry.add([("I_1#1", "1"), ("I_1#1", "2"), ("I_3#1", "3")])
    summary = registry.summary()
    assert registry.conflicting_entities() == {"I_1#1"}
    assert summary["membership_conflict_entities"] == 1
    assert summary["retained_single_membership_entities"] == 1
    assert summary["maximum_memberships_per_retained_entity"] == 1
    registry.close()


def test_kruskal_fdr_handles_all_identical_groups():
    frame = pd.DataFrame(
        {
            "category_label": ["a"] * 5 + ["b"] * 5,
            "metric": [0.0] * 10,
        }
    )
    tests, descriptions = kruskal_fdr(frame, ["metric"], "include_mixed", 5)
    assert len(descriptions) == 2
    assert tests.loc[0, "kruskal_h"] == 0.0
    assert tests.loc[0, "p_value"] == 1.0
    assert tests.loc[0, "test_status"] == "all_values_identical"
    assert tests.loc[0, "fdr_bh_p_value"] == 1.0
