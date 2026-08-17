"""Role-specific views derived from the first-order RefQ relation."""

from __future__ import annotations

import random
from typing import Any, Iterable, Optional

import networkx as nx
import pandas as pd
from script.complex_network_analysis.build_network.build_Graph import DG2G


def cross_project_edges(edges: pd.DataFrame) -> pd.DataFrame:
    return edges[~edges["is_self_loop"]].copy()


def directed_to_undirected_edges(
    edges: pd.DataFrame,
    source_col: str = "source_project_id",
    target_col: str = "target_project_id",
    weight_col: str = "weight",
    drop_self_loop: bool = True,
) -> pd.DataFrame:
    missing = [col for col in (source_col, target_col, weight_col) if col not in edges]
    if missing:
        raise KeyError(f"missing undirected-view columns: {', '.join(missing)}")
    work = edges[[source_col, target_col, weight_col]].copy()
    work[source_col] = work[source_col].astype(str)
    work[target_col] = work[target_col].astype(str)
    if drop_self_loop:
        work = work[work[source_col] != work[target_col]]

    directed_rows = (
        work.groupby([source_col, target_col], as_index=False)
        .agg(weight=(weight_col, "sum"), directed_edge_count=(weight_col, "size"))
    )
    directed = nx.DiGraph()
    for source, target, weight, edge_count in directed_rows.itertuples(index=False, name=None):
        directed.add_edge(
            source,
            target,
            weight=weight,
            multiplicity=int(edge_count),
        )

    # DG2G mutates edge attributes, so only this local graph is passed to it.
    undirected = DG2G(directed, multiplicity=True, double_self_loop=False)
    rows = []
    for node_a, node_b, data in undirected.edges(data=True):
        node_a, node_b = str(node_a), str(node_b)
        rows.append(
            {
                "node_u": min(node_a, node_b),
                "node_v": max(node_a, node_b),
                "weight": data["weight"],
                "directed_edge_count": int(data["multiplicity"]),
            }
        )

    if not rows:
        return pd.DataFrame(
            {
                "node_u": pd.Series(dtype="object"),
                "node_v": pd.Series(dtype="object"),
                "weight": directed_rows["weight"].iloc[:0],
                "directed_edge_count": pd.Series(dtype="int64"),
            }
        ).reset_index(drop=True)
    return pd.DataFrame(rows).sort_values(["node_u", "node_v"]).reset_index(drop=True)


def analyze_undirected_view(
    undirected_edges: pd.DataFrame,
    random_seed: int,
    brokerage_sample_size: int,
    node_ids: Optional[Iterable[object]] = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graph = nx.Graph()
    if node_ids is not None:
        graph.add_nodes_from(str(node) for node in node_ids)
    for row in undirected_edges.itertuples(index=False):
        graph.add_edge(str(row.node_u), str(row.node_v), weight=float(row.weight))
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    lcc_nodes = components[0] if components else set()
    lcc = graph.subgraph(lcc_nodes).copy()
    lcc_edges = undirected_edges[
        undirected_edges["node_u"].isin(lcc_nodes) & undirected_edges["node_v"].isin(lcc_nodes)
    ].copy()

    communities = list(nx.community.louvain_communities(lcc, weight="weight", seed=random_seed)) if lcc else []
    modularity = nx.community.modularity(lcc, communities, weight="weight") if communities else 0.0
    membership = {
        node: community_id
        for community_id, nodes in enumerate(sorted(communities, key=lambda values: (-len(values), min(values))))
        for node in nodes
    }
    community_frame = pd.DataFrame(
        [{"project_id": node, "community_id": community, "community_size": len(communities[community])}
         for node, community in sorted(membership.items())]
    )

    clustering = nx.clustering(lcc, weight=None) if lcc else {}
    if len(lcc) > 1:
        k = min(brokerage_sample_size, len(lcc))
        brokerage = nx.betweenness_centrality(lcc, k=k, normalized=True, seed=random_seed, weight=None)
    else:
        brokerage = {node: 0.0 for node in lcc}
    node_frame = pd.DataFrame(
        [
            {
                "project_id": str(node),
                "undirected_degree": int(lcc.degree(node)),
                "undirected_strength": float(lcc.degree(node, weight="weight")),
                "local_clustering": float(clustering.get(node, 0.0)),
                "betweenness_brokerage": float(brokerage.get(node, 0.0)),
                "community_id": membership.get(node),
            }
            for node in sorted(lcc)
        ]
    )
    brokerage_frame = node_frame.sort_values(
        ["betweenness_brokerage", "undirected_degree", "project_id"], ascending=[False, False, True]
    ).reset_index(drop=True)
    summary = {
        "view": "U(G_RefQ)",
        "operator_order": "first_order_undirected_view",
        "nodes": graph.number_of_nodes(),
        "edge_observed_nodes": sum(1 for node in graph if graph.degree(node) > 0),
        "undirected_edges": graph.number_of_edges(),
        "components": len(components),
        "isolates": nx.number_of_isolates(graph),
        "lcc_nodes": lcc.number_of_nodes(),
        "lcc_edges": lcc.number_of_edges(),
        "lcc_coverage": lcc.number_of_nodes() / graph.number_of_nodes() if graph else 0.0,
        "average_clustering_lcc": nx.average_clustering(lcc) if lcc else 0.0,
        "transitivity_lcc": nx.transitivity(lcc) if lcc else 0.0,
        "algorithmic_community_method": "networkx_louvain",
        "algorithmic_communities": len(communities),
        "modularity": float(modularity),
        "brokerage_method": "unweighted_approximate_betweenness",
        "brokerage_sample_size": min(brokerage_sample_size, len(lcc)),
        "random_seed": random_seed,
    }
    return summary, lcc_edges, community_frame, brokerage_frame
