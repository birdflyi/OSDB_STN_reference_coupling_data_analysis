#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/14 14:21
# @Author : 'Lou Zehua'
# @File   : build_gh_collab_net_pattern.py
import copy

import networkx as nx
import pandas as pd

from GH_CoRE.model.ER_config_parser import df_ref_tuples
from GH_CoRE.working_flow import get_repo_name_fileformat, get_repo_year_filename, read_csvs
from matplotlib import pyplot as plt

from script.complex_network_analysis.build_network.build_Graph import build_Graph
from script.complex_network_analysis.build_network.build_gh_collab_net import build_collab_net

plt.switch_backend('TkAgg')


def get_pattern_from_ER_config_df_ref_tuples(df_src_tar=None, use_all_nodes=True):
    df_src_tar = df_src_tar if df_src_tar is not None else df_ref_tuples
    if use_all_nodes:  # init all nodes based on the network pattern dataframe df_ref_tuples with only_build_nodes=True
        base_graph = build_Graph(df_src_tar[['source_node_type', 'target_node_type', 'event_type']], base_graph=None,
                                default_node_types=['__repr__', '__repr__'], node_type_canopy=False,
                                use_df_col_as_default_type=True, w_trunc=1, out_g_type='MDG', only_build_nodes=True)
    else:
        base_graph = None

    G_pattern = build_Graph(df_src_tar[['source_node_type', 'target_node_type', 'event_type']],
                            base_graph=base_graph, default_node_types=['__repr__', '__repr__'], node_type_canopy=False,
                            default_edge_type="event_type", edge_type_canopy=True, use_df_col_as_default_type=True,
                            w_trunc=1, out_g_type='DG')
    return G_pattern


def get_pattern_from_G(G, directed=True):
    G_pattern = nx.DiGraph()

    # init default_node_pattern_attr and default_edge_pattern_attr
    default_node_pattern_attr = {
        'node_type': None,
        'weight': 0,
        'graph_accumulate_weight': 0,
        'graph_accumulate_multiplicity': 0,
    }
    default_edge_pattern_attr = {
        'edge_type': {},
        'weight': 0,
        'graph_accumulate_weight': 0,
        'graph_accumulate_multiplicity': 0,
    }

    # count weight and multiplicity
    for u, e_dict in G.nodes(data=True):
        u_node_pattern = e_dict.get('node_type', 'default')
        node_pattern_attr = copy.deepcopy(default_node_pattern_attr)
        if u_node_pattern not in G_pattern.nodes():
            G_pattern.add_node(u_node_pattern, **node_pattern_attr)
        G_pattern.nodes[u_node_pattern]['node_type'] = u_node_pattern
        G_pattern.nodes[u_node_pattern]['weight'] += 1
        G_pattern.nodes[u_node_pattern]['graph_accumulate_weight'] += e_dict.get('weight', 1)
        G_pattern.nodes[u_node_pattern]['graph_accumulate_multiplicity'] += e_dict.get('multiplicity', 1)

    for u, v, e_dict in G.edges(data=True):
        u_type = G.nodes[u].get('node_type', 'default')
        v_type = G.nodes[v].get('node_type', 'default')
        edge_pattern_attr = copy.deepcopy(default_edge_pattern_attr)
        if directed:
            if (u_type, v_type) not in G_pattern.edges:
                G_pattern.add_edge(u_type, v_type, **edge_pattern_attr)
        else:
            if (u_type, v_type) not in G_pattern.edges and (v_type, u_type) not in G_pattern.edges:
                G_pattern.add_edge(u_type, v_type, **edge_pattern_attr)
            if (u_type, v_type) not in G_pattern.edges:
                (u_type, v_type) = (v_type, u_type)
        uv_edge_type = e_dict.get('edge_type', 'default')

        if isinstance(uv_edge_type, dict):
            for k_edge_type, v_edge_type in uv_edge_type.items():
                if bool(v_edge_type):
                    G_pattern.edges[u_type, v_type]['edge_type'][k_edge_type] = True
        elif isinstance(uv_edge_type, str):
            G_pattern.edges[u_type, v_type]['edge_type'][uv_edge_type] = True
        else:
            raise TypeError("uv_edge_type in edge attributes must be a type str or a truth dict of edge types.")
        G_pattern.edges[u_type, v_type]['weight'] += 1
        G_pattern.edges[u_type, v_type]['graph_accumulate_weight'] += e_dict.get('weight', 1)
        G_pattern.edges[u_type, v_type]['graph_accumulate_multiplicity'] += e_dict.get('multiplicity', 1)

    return G_pattern


if __name__ == '__main__':
    import os

    from etc import filePathConf

    df_ref_tuples.to_csv(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                      'analysis_results/df_ref_tuples.csv'))

    USE_GLOBAL_G_PATTERN = True
    if USE_GLOBAL_G_PATTERN:
        # edge type filters
        use_relation_types = ["EventAction", "Reference"][:1]
        df_ref_tuples_edge_filtered = df_ref_tuples[df_ref_tuples.apply(lambda s: s["relation_type"] in use_relation_types, axis=1)]
        df_ref_tuples_edge_filtered.to_csv(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                                        'analysis_results/df_ref_tuples_edge_filtered.csv'))
        G_pattern = get_pattern_from_ER_config_df_ref_tuples(df_ref_tuples_edge_filtered)
        for n in G_pattern.nodes(data=True):
            print(n)
        for e in G_pattern.edges(data=True):
            print(e)

        node_type_set = set(df_ref_tuples["source_node_type"]).union(set(df_ref_tuples["target_node_type"]))
        node_types = list(node_type_set)
    else:
        repo_names = ["TuGraph-family/tugraph-db", "facebook/rocksdb", "cockroachdb/cockroach"][0:1]
        year = 2023
        relation_extraction_save_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                                  'repos_GH_CoRE')
        filenames_exists = os.listdir(relation_extraction_save_dir)
        if repo_names:
            repo_names_fileformat = list(map(get_repo_name_fileformat, repo_names))
            filenames = [get_repo_year_filename(s, year) for s in repo_names_fileformat]
            filenames = [filename for filename in filenames if filename in filenames_exists]
        else:
            filenames = filenames_exists

        df_dbms_repos_dict = read_csvs(relation_extraction_save_dir, filenames=filenames, index_col=None)
        # test for 1 repo case
        repo_keys = list(df_dbms_repos_dict.keys())
        df_dbms_repo = df_dbms_repos_dict[repo_keys[0]]
        G_repo = build_collab_net(df_dbms_repo, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
                         default_node_types=['src_entity_type', 'tar_entity_type'], default_edge_type="event_type",
                         init_record_as_edge_attrs=True, use_df_col_as_default_type=True, out_g_type='DG')
        G_pattern = get_pattern_from_G(G_repo, directed=True)
        for u, e_dict in G_pattern.nodes(data=True):
            print(G_pattern.nodes[u]['node_type'], e_dict)

        for u, v, e_dict in G_pattern.edges(data=True):
            print(G_pattern.nodes[u]['node_type'], G_pattern.nodes[v]['node_type'], e_dict)

        node_types = list(G_pattern.nodes)

    pos = nx.kamada_kawai_layout(G_pattern)
    # 使用matplotlib的colormap生成不同的颜色
    cmap = plt.get_cmap('jet', 15)
    len_node_types = len(node_types)
    colors = [cmap(i) for i in range(len_node_types)]
    color_map = dict(zip(node_types, colors[:len_node_types]))
    node_size_map = dict(zip(node_types, [200] * len_node_types))
    df_nodes_data = pd.DataFrame(dict(G_pattern.nodes(data=True)))
    df_nodes_type = df_nodes_data.loc["node_type"]
    node_color = df_nodes_type.apply(lambda x: color_map[x]).values
    node_size = df_nodes_type.apply(lambda x: node_size_map[x]).values
    node_labels = nx.get_node_attributes(G_pattern, 'node_type')
    edge_labels = nx.get_edge_attributes(G_pattern, 'weight')

    if USE_GLOBAL_G_PATTERN:
        width = list(nx.get_edge_attributes(G_pattern, 'weight').values())
    else:
        # the wight values are too large if USE_GLOBAL_G_PATTERN = False, use bool or log to get a proper width list
        width = list(map(bool, nx.get_edge_attributes(G_pattern, 'weight').values()))
    nx.draw(G_pattern, pos, labels=node_labels, node_size=node_size, node_color=node_color, edge_color="gray",
            width=width, font_size=6)
    nx.draw_networkx_edge_labels(G_pattern, pos, edge_labels=edge_labels, font_size=6)

    plt.title('GH Collab Network Pattern', fontsize=15)
    plt.savefig(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                             "analysis_results/HIN_pattern_GH_Collab_EAct_scale1_trunc1.png"), format="PNG")
    plt.show()
