import networkx as nx
import pandas as pd

from script.ch5_reference_quotient import network_views
from script.ch5_reference_quotient.network_views import (
    analyze_undirected_view,
    directed_to_undirected_edges,
)


def test_u_g_refq_merges_reciprocal_edges_and_excludes_self_loop():
    edges = pd.DataFrame(
        [
            {"source_project_id": "1", "target_project_id": "2", "weight": 2},
            {"source_project_id": "2", "target_project_id": "1", "weight": 3},
            {"source_project_id": "1", "target_project_id": "1", "weight": 7},
        ]
    )
    undirected = directed_to_undirected_edges(edges)
    assert undirected.to_dict("records") == [
        {"node_u": "1", "node_v": "2", "weight": 5, "directed_edge_count": 2}
    ]


def test_u_g_refq_preserves_duplicate_counts_self_loops_and_input():
    edges = pd.DataFrame(
        [
            {"source": "b", "target": "a", "weight": 2},
            {"source": "b", "target": "a", "weight": 4},
            {"source": "a", "target": "b", "weight": 3},
            {"source": "a", "target": "a", "weight": 7},
        ]
    )
    original = edges.copy(deep=True)

    result = directed_to_undirected_edges(
        edges,
        source_col="source",
        target_col="target",
        weight_col="weight",
        drop_self_loop=False,
    )

    assert result.to_dict("records") == [
        {"node_u": "a", "node_v": "a", "weight": 7, "directed_edge_count": 1},
        {"node_u": "a", "node_v": "b", "weight": 9, "directed_edge_count": 3},
    ]
    pd.testing.assert_frame_equal(edges, original)


def test_u_g_refq_calls_dg2g_with_refq_semantics(monkeypatch):
    calls = []

    def fake_dg2g(graph, **kwargs):
        calls.append((graph.copy(), kwargs))
        return nx.Graph(graph)

    monkeypatch.setattr(network_views, "DG2G", fake_dg2g)
    result = directed_to_undirected_edges(
        pd.DataFrame([{"source": "a", "target": "b", "weight": 3}]),
        source_col="source",
        target_col="target",
        weight_col="weight",
    )

    assert result.to_dict("records") == [
        {"node_u": "a", "node_v": "b", "weight": 3, "directed_edge_count": 1}
    ]
    assert calls[0][1] == {"multiplicity": True, "double_self_loop": False}


def test_u_g_refq_empty_edges_preserve_output_schema():
    result = directed_to_undirected_edges(
        pd.DataFrame(columns=["source", "target", "weight"]),
        source_col="source",
        target_col="target",
        weight_col="weight",
    )

    assert list(result.columns) == ["node_u", "node_v", "weight", "directed_edge_count"]
    assert result.empty
    assert isinstance(result.index, pd.RangeIndex)
    assert str(result["directed_edge_count"].dtype) == "int64"


def test_u_g_refq_preserves_explicit_isolated_node_domain():
    undirected = pd.DataFrame(
        [{"node_u": "1", "node_v": "2", "weight": 5, "directed_edge_count": 1}]
    )
    summary, _, _, _ = analyze_undirected_view(
        undirected,
        random_seed=7,
        brokerage_sample_size=10,
        node_ids=["1", "2", "3"],
    )
    assert summary["nodes"] == 3
    assert summary["edge_observed_nodes"] == 2
    assert summary["isolates"] == 1
    assert summary["components"] == 2
