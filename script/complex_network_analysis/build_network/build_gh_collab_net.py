#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/15 13:53
# @Author : 'Lou Zehua'
# @File   : build_gh_collab_net.py
import os

import networkx as nx
import numpy as np
import pandas as pd

from GH_CoRE.working_flow import read_csvs
from GH_CoRE.working_flow.query_OSDB_github_log import get_repo_name_fileformat, get_repo_year_filename

from etc import filePathConf
from script.complex_network_analysis.build_network.build_Graph import build_Graph


if __name__ == '__main__':
    repo_names = ['opengauss-mirror/openGauss-server', "TuGraph-family/tugraph-db", "facebook/rocksdb", "cockroachdb/cockroach"][0:1]
    year = 2023
    relation_extraction_save_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                              'GitHub_Collaboration_Network_repos')
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
    G_repo = build_Graph(df_dbms_repo, base_graph=None, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
                         src_node_attrs=None, tar_node_attrs=None,
                         init_node_weight=True, nt_key_in_attr="node_type", default_node_types=['src_entity_type', 'tar_entity_type'], node_type_canopy=False,
                         edge_attrs=pd.Series(df_dbms_repo.to_dict("records")),
                         init_edge_weight=True, et_key_in_attr="edge_type", default_edge_type="event_type", edge_type_canopy=True,
                         attrs_is_shared_key_pdSeries=False, use_df_col_as_default_type=True, w_trunc=1, out_g_type='DG')

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt  # matplotlib.use必须在本句执行前运行

    pos = nx.spring_layout(G_repo, seed=1)

    df_nodes_data = pd.DataFrame(dict(G_repo.nodes(data=True)))
    df_nodes_type = df_nodes_data.loc["node_type"]
    node_type_set = set(df_nodes_type)
    cmap = plt.get_cmap('jet', 15)
    len_node_types = len(node_type_set)
    colors = [cmap(i % 15) for i in range(len_node_types)]
    color_map = dict(zip(node_type_set, colors[:len_node_types]))

    node_color = df_nodes_type.apply(lambda x: color_map[x]).values
    # node_size = df_nodes_data.apply(lambda x: x["weight"]*200 if pd.notna(x["weight"]) and isinstance(x["weight"], (int, float)) else 10).values
    # node_size_map = dict(zip(node_type_set, [200] * len_node_types))
    node_size = pd.DataFrame(G_repo.degree, columns=["entity_id", "degree"])["degree"].values

    node_labels = nx.get_node_attributes(G_repo, 'node_type')
    edge_labels = nx.get_edge_attributes(G_repo, 'weight')

    nx.draw_networkx_edge_labels(G_repo, pos, edge_labels=edge_labels)
    nx.draw(G_repo, pos, labels=node_labels, font_size=5, node_size=node_size, node_color=node_color, edge_color="black")

    plt.title(f'Graph of Repo {repo_keys[0]}', fontsize=15)
    plt.savefig(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                             f"analysis_results/collab_net_{repo_keys[0]}_scale1_trunc1.png"), format="PNG")
    plt.show()
