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
    repo_names = ["TuGraph-family/tugraph-db", "facebook/rocksdb", "cockroachdb/cockroach"][0:1]
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
    cnt = 0
    limit = 10
    for n in G_repo.nodes(data=True):
        print(n)
        cnt += 1
        if cnt > limit:
            break
    cnt = 0
    for e in G_repo.edges(data=True):
        print(e)
        cnt += 1
        if cnt > limit:
            break
