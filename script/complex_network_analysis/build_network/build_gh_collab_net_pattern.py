#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/14 14:21
# @Author : 'Lou Zehua'
# @File   : build_gh_collab_net_pattern.py

import networkx as nx
import numpy as np
import pandas as pd

from GH_CoRE.model.ER_config_parser import df_ref_tuples
from matplotlib import pyplot as plt

from script.complex_network_analysis.build_network.build_Graph import build_Graph

plt.switch_backend('TkAgg')

use_all_nodes = True
if use_all_nodes:
    base_graph = build_Graph(df_ref_tuples[['source_node_type', 'target_node_type', 'event_type']], base_graph=None,
                            default_node_types=['__repr__', '__repr__'], node_type_canopy=False,
                            use_df_col_as_default_type=True, w_trunc=1, out_g_type='MDG', only_build_nodes=True)
else:
    base_graph = None

use_relation_types = ["EventAction", "Reference"][:1]
df_ref_tuples_edge_filtered = df_ref_tuples[df_ref_tuples.apply(lambda s: s["relation_type"] in use_relation_types, axis=1)]
G_pattern = build_Graph(df_ref_tuples_edge_filtered[['source_node_type', 'target_node_type', 'event_type']], base_graph=base_graph,
                        default_node_types=['__repr__', '__repr__'], node_type_canopy=False,
                        edge_attrs='event_type', default_edge_type="event_type", edge_type_canopy=True,
                        use_df_col_as_default_type=True, w_trunc=1, out_g_type='DG')


if __name__ == '__main__':
    import os

    from etc import filePathConf

    for n in G_pattern.nodes(data=True):
        print(n)
    for e in G_pattern.edges(data=True):
        print(e)
    df_ref_tuples.to_csv(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                      'analysis_results/df_ref_tuples.csv'))
    df_ref_tuples_edge_filtered.to_csv(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                                    'analysis_results/df_ref_tuples_edge_filtered.csv'))

    node_type_set = set(df_ref_tuples["source_node_type"]).union(set(df_ref_tuples["target_node_type"]))
    node_types = list(node_type_set)
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

    nx.draw(G_pattern, pos, labels=node_labels, node_size=node_size, node_color=node_color, edge_color="gray",
            width=list(nx.get_edge_attributes(G_pattern, 'weight').values()), font_size=6)
    nx.draw_networkx_edge_labels(G_pattern, pos, edge_labels=edge_labels, font_size=6)

    plt.title('GH Collab Network Pattern', fontsize=15)
    plt.savefig(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                             "analysis_results/HIN_pattern_GH_Collab_EAct_scale1_trunc1.png"), format="PNG")
    plt.show()
