#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/4/19 21:14
# @Author : 'Lou Zehua'
# @File   : main.py

import os
import sys

import pandas as pd

from GH_CoRE.working_flow import get_repo_name_fileformat, get_repo_year_filename, read_csvs

from script.complex_network_analysis.Network_params_analysis import get_graph_feature
from script.complex_network_analysis.build_network.build_gh_collab_net import build_collab_net

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = cur_dir  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

from etc import filePathConf
from script.build_dataset.collaboration_relation_extraction import collaboration_relation_extraction_service


if __name__ == '__main__':
    year = 2023
    repo_names = None
    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]
    dbms_repos_dedup_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_DEDUP_CONTENT_DIR]
    collaboration_relation_extraction_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_GH_CORE_DIR]
    start_step = 2
    if start_step <= 1:
        collaboration_relation_extraction_service(dbms_repos_key_feats_path, dbms_repos_raw_content_dir,
                                                  dbms_repos_dedup_content_dir, collaboration_relation_extraction_dir,
                                                  repo_names=repo_names, stop_repo_names=None, year=year)
    if start_step <= 2:
        repo_names = None
        # repo_names = ["TuGraph-family/tugraph-db", "neo4j/neo4j", "facebook/rocksdb", "cockroachdb/cockroach"][0:2]
        relation_extraction_save_dir = collaboration_relation_extraction_dir
        filenames_exists = os.listdir(relation_extraction_save_dir)
        if repo_names:
            repo_names_fileformat = list(map(get_repo_name_fileformat, repo_names))
            filenames = [get_repo_year_filename(s, year) for s in repo_names_fileformat]
            filenames = [filename for filename in filenames if filename in filenames_exists]
        else:
            filenames = filenames_exists

        df_dbms_repos_dict_tmp = read_csvs(relation_extraction_save_dir, filenames=filenames, index_col=None)
        df_dbms_repos_dict = {k: df_dbms_repos_dict_tmp[k] for k in sorted(df_dbms_repos_dict_tmp)}
        g_feat_path = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                   'analysis_results/df_g_feat.csv')
        repo_keys = list(df_dbms_repos_dict.keys())
        if os.path.isfile(g_feat_path):
            df_g_feat_repo_key_as_index = pd.read_csv(g_feat_path, header="infer", index_col=None)
            repos_feat_dict_values = df_g_feat_repo_key_as_index.to_dict(orient="records")
            repos_feat_dict = {repos_feat_dict_values[k].get("repo_name", str(k)): repos_feat_dict_values[k]
                               for k in range(len(repos_feat_dict_values))}
        else:
            repos_feat_dict = {}

        for repo_key in repo_keys:
            if repo_key in repos_feat_dict.keys():
                continue

            graph_feature_record = {"repo_name": repo_key}
            df_dbms_repo = df_dbms_repos_dict[repo_key]
            G = build_collab_net(df_dbms_repo, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
                                 default_node_types=['src_entity_type', 'tar_entity_type'],
                                 default_edge_type="event_type",
                                 init_record_as_edge_attrs=True, use_df_col_as_default_type=True, out_g_type='G')
            graph_feature_record_complex_network = get_graph_feature(G)
            graph_feature_record.update(graph_feature_record_complex_network)
            repos_feat_dict[repo_key] = graph_feature_record
            df_g_feat = pd.DataFrame.from_dict(repos_feat_dict, orient='index')
            df_g_feat.to_csv(g_feat_path, index=False)
            print(f"{repo_key} saved into df_g_feat!")
