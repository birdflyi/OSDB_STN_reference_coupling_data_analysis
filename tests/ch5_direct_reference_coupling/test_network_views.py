import pandas as pd

from script.ch5_direct_reference_coupling.network_views import directed_to_undirected_edges
from script.ch5_direct_reference_coupling.rq2a_source_view import summarize_source_view
from script.ch5_direct_reference_coupling.rq2b_target_view import summarize_target_view


def test_directed_to_undirected_merges_reciprocal_edges_and_drops_self_loop():
    edges = pd.DataFrame(
        [
            {"source_repo": "a", "target_repo": "b", "weight": 2},
            {"source_repo": "b", "target_repo": "a", "weight": 3},
            {"source_repo": "a", "target_repo": "a", "weight": 7},
        ]
    )

    undirected = directed_to_undirected_edges(edges)

    assert len(undirected) == 1
    assert undirected.iloc[0].to_dict() == {
        "node_u": "a",
        "node_v": "b",
        "weight": 5,
        "directed_edge_count": 2,
    }


def test_source_and_target_views_have_separate_roles():
    edges = pd.DataFrame(
        [
            {"source_repo": "seed/a", "target_repo": "target/x", "weight": 2},
            {"source_repo": "seed/a", "target_repo": "target/y", "weight": 1},
            {"source_repo": "seed/b", "target_repo": "target/x", "weight": 4},
        ]
    )

    source = summarize_source_view(edges)
    target = summarize_target_view(edges)

    assert source.loc[source["source_repo"] == "seed/a", "out_degree"].iloc[0] == 2
    assert source.loc[source["source_repo"] == "seed/a", "out_strength"].iloc[0] == 3
    assert target.loc[target["target_repo"] == "target/x", "in_degree"].iloc[0] == 2
    assert target.loc[target["target_repo"] == "target/x", "in_strength"].iloc[0] == 6
