import pandas as pd

from script.ch5_direct_reference_coupling.edge_table import build_direct_edge_table
from script.ch5_direct_reference_coupling.reference_filtering import (
    deduplicate_references,
    drop_unresolved_targets,
    filter_direct_references,
)


def test_reference_filter_dedup_and_edge_aggregation_direction():
    records = pd.DataFrame(
        [
            {"event_id": "e1", "relation_type": "Reference", "source_repo": "seed/a", "target_repo": "target/x"},
            {"event_id": "e1", "relation_type": "Reference", "source_repo": "seed/a", "target_repo": "target/x"},
            {"event_id": "e2", "relation_type": "Reference", "source_repo": "seed/a", "target_repo": "target/x"},
            {"event_id": "e3", "relation_type": "EventAction", "source_repo": "seed/a", "target_repo": "target/y"},
            {"event_id": "e4", "relation_type": "Reference", "source_repo": "target/x", "target_repo": "seed/a"},
        ]
    )

    refs = filter_direct_references(records)
    deduped = deduplicate_references(refs)
    edges = build_direct_edge_table(deduped, analysis_seed_repos=["seed/a"])

    ax = edges[(edges["source_repo"] == "seed/a") & (edges["target_repo"] == "target/x")].iloc[0]
    xa = edges[(edges["source_repo"] == "target/x") & (edges["target_repo"] == "seed/a")].iloc[0]
    assert ax["weight"] == 2
    assert xa["weight"] == 1
    assert bool(ax["source_is_seed"]) is True
    assert bool(ax["target_is_seed"]) is False


def test_unresolved_target_drop_and_self_loop_flag():
    records = pd.DataFrame(
        [
            {"relation_type": "Reference", "source_repo": "seed/a", "target_repo": ""},
            {"relation_type": "Reference", "source_repo": "seed/a", "target_repo": "seed/a"},
        ]
    )

    resolved = drop_unresolved_targets(records)
    edges = build_direct_edge_table(resolved, analysis_seed_repos=["seed/a"], drop_self_loop=False)

    assert len(resolved) == 1
    assert len(edges) == 1
    assert bool(edges.iloc[0]["is_self_loop"]) is True
