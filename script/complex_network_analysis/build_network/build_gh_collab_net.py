#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/15 13:53
# @Author : 'Lou Zehua'
# @File   : build_gh_collab_net.py
import os

import networkx as nx
import pandas as pd

from GH_CoRE.working_flow import read_csvs
from GH_CoRE.working_flow.query_OSDB_github_log import get_repo_name_fileformat, get_repo_year_filename

from script.complex_network_analysis.build_network.build_Graph import build_Graph


def build_collab_net(df_src_tar, src_tar_colnames=None, base_graph=None, default_node_types=None,
                     default_edge_type=None, init_record_as_edge_attrs=True, attrs_is_shared_key_pdSeries=False,
                     use_df_col_as_default_type=True, out_g_type='DG'):
    """
    :param df_src_tar: type pd.DataFrame, a dataframe within source node column and target node column.
    :param src_tar_colnames: the source node column name and target node column name in df_src_tar.columns
    :param base_graph: a base graph can be parsed into nx.MultiGraph type
    :param default_node_types: set a pair of default node types when there is no "node_type" key in node_attrs.
        The "node_type" in src_node_attrs and tar_node_attrs always take effect first.
        default_node_types=[default_src_node_type_setting, default_tar_node_type_setting] each node type settings can be:
            None: for no default node types, the key "node_type" will not be generated automatically.
            "__repr__": use the value str of node as default node type.
            name_str: name_str is a string that is not "__repr__".
                if use_df_col_as_default_type=True, try to get the series df_src_tar[name_str] as the default node type series;
                if use_df_col_as_default_type=False, use the name_str as default node type to extend a default node type series.
    :param default_edge_type: set default edge type when there is no "edge_type" key in edge_attrs.
        The "edge_type" in edge_attrs always take effect first.
        default_edge_type=None: for no default edge type, the key "edge_type" will not be generated automatically.
        default_edge_type="__repr__": use SrcNodeType_TarNodeType pattern as default edge type for each edge.
        default_edge_type=name_str: name_str is a string that is not "__repr__".
            if use_df_col_as_default_type=True, try to get the series df_src_tar[name_str] as the default edge type series;
            if use_df_col_as_default_type=False, use the name_str as default edge type to extend a default edge type series.
    :param init_record_as_edge_attrs: init the record info as edge_attrs for each edge.
    :param attrs_is_shared_key_pdSeries: the parsing format of df_src_tar. The types of src_node_attrs, tar_node_attrs,
        edge_attrs are pd.Series if True else list(dict) or pd.Series(dict) records.
    :param use_df_col_as_default_type: if True, default_node_types and default_edge_type settings can use a name_str to
        get the series df_src_tar[name_str] as the default node/edge type series.
    :param out_g_type: the return type of result graph, should be in ['MDG', 'DG', 'G'].
    :return: G: default type out_g_type='DG'
    """
    src_tar_colnames = src_tar_colnames or ['src_entity_id', 'tar_entity_id']
    default_node_types = default_node_types or ['src_entity_type', 'tar_entity_type']
    default_edge_type = default_edge_type or "event_type"
    edge_attrs = pd.Series(df_src_tar.to_dict("records")) if init_record_as_edge_attrs else None
    G = build_Graph(df_src_tar, base_graph=base_graph, src_tar_colnames=src_tar_colnames,
                    src_node_attrs=None, tar_node_attrs=None, init_node_weight=True, nt_key_in_attr="node_type",
                    default_node_types=default_node_types, node_type_canopy=False,
                    edge_attrs=edge_attrs, init_edge_weight=True, et_key_in_attr="edge_type",
                    default_edge_type=default_edge_type, edge_type_canopy=True,
                    attrs_is_shared_key_pdSeries=attrs_is_shared_key_pdSeries,
                    use_df_col_as_default_type=use_df_col_as_default_type, w_trunc=1, out_g_type=out_g_type)
    return G


if __name__ == '__main__':
    from etc import filePathConf
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
    G_repo = build_collab_net(df_dbms_repo, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
                     default_node_types=['src_entity_type', 'tar_entity_type'], default_edge_type="event_type",
                     init_record_as_edge_attrs=True, use_df_col_as_default_type=True, out_g_type='DG')

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
