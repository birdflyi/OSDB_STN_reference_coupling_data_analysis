#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2026/2/1 5:53
# @Author : 'Lou Zehua'
# @File   : dg_dbms_repos_ref_net_vis.py

import os
import sys

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = os.path.dirname(cur_dir)  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

import logging
import pickle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # matplotlib.use必须在本句执行前运行
import networkx as nx
import numpy as np
import pandas as pd

from GH_CoRE.utils.logUtils import setup_logging
from etc import filePathConf
from reference_descriptive_analysis import select_target_repos

setup_logging(base_dir=pkg_rootdir)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    flag_skip_existing_files = True
    year = 2023
    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    df_target_repos = select_target_repos(dbms_repos_key_feats_path, year, re_preprocess=False, ret="dataframe")
    use_repo_nodes_only = True
    homo_dg_dbms_repos_ref_net_node_agg_filename = "homo_dg_dbms_repos_ref_net_node_agg.gexf"
    graph_network_dir = filePathConf.absPathDict[filePathConf.GRAPH_NETWORK_DIR]
    homo_dg_dbms_repos_ref_net_node_agg_path = os.path.join(graph_network_dir,
                                                            homo_dg_dbms_repos_ref_net_node_agg_filename)
    G_repo = nx.read_gexf(homo_dg_dbms_repos_ref_net_node_agg_path)
    logger.info(f"Load {homo_dg_dbms_repos_ref_net_node_agg_path}...")
    # add node attributes: "degree", "repo_name"
    degrees = dict(G_repo.degree())
    nx.set_node_attributes(G_repo, degrees, 'degree')

    df_target_repos["repo_id"] = df_target_repos["repo_id"].astype(str)
    df_filtered = df_target_repos.dropna(subset=['repo_id'])  # 去除key列为空值的行
    df_filtered = df_filtered.drop_duplicates(subset=['repo_id'], keep='first')  # 去除key列重复值的行，保留第一次出现的
    # show the repo_name in df_target_repos as node labels
    repo_id_name_dict = df_filtered.set_index('repo_id')['repo_name'].to_dict()
    for node in G_repo.nodes():
        if str(node).startswith("R_"):
            repo_id = node.split("_")[1]
            G_repo.nodes[node]['repo_name'] = repo_id_name_dict.get(repo_id, "")

    # filter nodes and edges
    only_dbms_repo = True
    drop_self_loop = True
    node_w_threshold = 1
    edge_w_threshold = 1
    if only_dbms_repo:
        rm_nodes = [n for n, d in G_repo.nodes.items() if d["degree"] < node_w_threshold or not d.get("repo_name")]
    else:
        rm_nodes = [n for n, d in G_repo.nodes.items() if d["degree"] < node_w_threshold]
    G_repo.remove_nodes_from(rm_nodes)

    if drop_self_loop:
        rm_edges = [(u, v) for u, v, d in G_repo.edges(data=True) if d['weight'] < edge_w_threshold or u == v]
    else:
        rm_edges = [(u, v) for u, v, d in G_repo.edges(data=True) if d['weight'] < edge_w_threshold]
    G_repo.remove_edges_from(rm_edges)
    # largest_wcc = max(nx.weakly_connected_components(G_repo), key=len)
    # G_repo = G_repo.subgraph(largest_wcc)

    # 设置随机种子以确保复现性
    seed = 42
    np.random.seed(seed)
    logger.info(f"Calculate networkx position with seed={seed}...")
    layout_dir = os.path.join(graph_network_dir, 'visualization')
    layout_filename = 'spring_layout_only_dbms_repo.pkl' if only_dbms_repo else 'spring_layout.pkl'
    layout_path = os.path.join(layout_dir, layout_filename)
    if not flag_skip_existing_files or not os.path.exists(layout_path):
        pos = nx.spring_layout(G_repo, seed=seed)
        with open(layout_path, 'wb') as f:
            pickle.dump(pos, f)
        logger.info(f"{layout_path} saved!")
    else:
        with open(layout_path, 'rb') as f:
            pos = pickle.load(f)
        logger.info(f"{layout_path} loaded!")

    node_size = pd.DataFrame(G_repo.degree, columns=["entity_id", "degree"])["degree"].values
    # set node_color
    df_nodes_data = pd.DataFrame(dict(G_repo.nodes(data=True)))
    df_nodes_type = df_nodes_data.loc["node_type"]
    node_type_set = set(df_nodes_type)
    cmap = plt.get_cmap('jet', 15)
    len_node_types = len(node_type_set)
    colors = [cmap(i % 15) for i in range(len_node_types)]
    color_map = dict(zip(node_type_set, colors[:len_node_types]))
    node_color = df_nodes_type.apply(lambda x: color_map[x]).values
    # set node labels and edge labels
    node_labels = nx.get_node_attributes(G_repo, 'repo_name')
    edge_labels = nx.get_edge_attributes(G_repo, 'weight')
    edge_labels = {k: int(v) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G_repo, pos, font_size=5, edge_labels=edge_labels)
    nx.draw(G_repo, pos, node_size=node_size, node_color=node_color, labels=node_labels, font_size=5, edge_color="gray")

    plt.title(f'Homogeneous DBMS Repos Reference Network(edge_weight >= {edge_w_threshold})', fontsize=15)
    plt.savefig(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                             f"analysis_results/homo_only_dbms_repos_ref_net_scale1_trunc1__edge_w_gt_{edge_w_threshold}.png"), format="PNG")
    plt.show()
