#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2026/1/17 17:41
# @Author : 'Lou Zehua'
# @File   : granular_aggregation.py

import logging
import os
import traceback
from functools import partial

import pandas as pd
from GH_CoRE import columns_simple
from GH_CoRE.model.Relation_extraction import get_obj_collaboration_tuples_from_record, get_df_collaboration, \
    save_GitHub_Collaboration_Network
from GH_CoRE.utils import QueryCache
from GH_CoRE.utils.logUtils import setup_logging
from GH_CoRE.working_flow.body_content_preprocessing import read_csvs, dedup_content
from GH_CoRE.working_flow.query_OSDB_github_log import query_repo_log_each_year_to_csv_dir

from etc import filePathConf
from script import pkg_rootdir
from script.build_dataset.query_repos_event_log import query_OSDB_github_log_from_dbserver
from script.build_dataset.repo_filter import get_filenames_by_repo_names, get_intersection

setup_logging(base_dir=pkg_rootdir)
logger = logging.getLogger(__name__)


def nt_granu_agg(row, level='repo'):
    row = pd.Series(row)
    if row["tar_entity_match_pattern_type"] in ["GitHub_Service_External_Links", "GitHub_Other_Service", ""]:
        pass  # no repo id
    elif not row["tar_entity_objnt_prop_dict"]:
        pass  # no identifiable features
    else:
        tar_entity_objnt_prop_dict = row["tar_entity_objnt_prop_dict"]
        tar_entity_type = row["tar_entity_type"]
        if level == 'repo':
            # "Object": GitHub_Files_FileChanges GitHub_Other_Links
            if 'repo_id' in row["tar_entity_match_pattern_type"]:
                if tar_entity_type == 'repo':  # appropriate granularity
                    pass
                tar_entity_id = tar_entity_objnt_prop_dict['repo_id']
                tar_entity_type =
            if tar_entity_type in ["Issue", "PullRequest"]:
            if tar_entity_type in row["tar_entity_match_pattern_type"]:
                tar_entity_id = tar_entity_objnt_prop_dict[tar_entity_type.lower() + '_id']
    return


if __name__ == '__main__':
    year = 2023
    # dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    # dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]
    # dbms_repos_dedup_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_DEDUP_CONTENT_DIR]
    collaboration_relation_extraction_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_GH_CORE_DIR]
    repo_names = ["pingcap/tidb", "tikv/tikv"]
    filenames = get_filenames_by_repo_names(repo_names, year)
    df_dbms_repos_dict = read_csvs(collaboration_relation_extraction_dir, filenames=filenames, index_col=None)
    df_dbms_repo = df_dbms_repos_dict[filenames[0]]
    # relation filter
    df_dbms_repo = df_dbms_repo[df_dbms_repo["relation_type"] == "Reference"]
    # target node granular aggregation
    df_dbms_repo = df_dbms_repo.apply(nt_granu_agg)
    # G_repo = build_collab_net(df_dbms_repo, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
    #                           default_node_types=['src_entity_type', 'tar_entity_type'], default_edge_type="event_type",
    #                           init_record_as_edge_attrs=True, use_df_col_as_default_type=True, out_g_type='DG')
