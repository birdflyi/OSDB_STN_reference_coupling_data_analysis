#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2026/1/10 15:01
# @Author : 'Lou Zehua'
# @File   : reference_descriptive_analysis.py

import os
import sys

from script.build_dataset.collaboration_relation_extraction import process_body_content, \
    collaboration_relation_extraction
from script.build_dataset.repo_filter import get_filenames_by_repo_names
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

import logging
import pandas as pd

from GH_CoRE.data_dict_settings import columns_simple, body_columns_dict, event_columns_dict, re_ref_patterns
from GH_CoRE.utils import get_params_condition
from GH_CoRE.utils.conndb import ConnDB
from GH_CoRE.utils.logUtils import setup_logging
from GH_CoRE.working_flow import query_repo_log_each_year_to_csv_dir, read_csvs

from etc import filePathConf
from script.utils.validate import ValidateFunc, complete_license_info, complete_github_repo_id

setup_logging(base_dir=pkg_rootdir)
logger = logging.getLogger(__name__)


# 步骤2: 数据收集与预处理
def select_target_repos(dbms_repos_key_feats_path, year=2023, re_preprocess=False):
    """
    :dbms_repos_key_feats_path 从dbdb.io与dbengines收录列表中选择符合条件的DBMS项目
    :re_preprocess set True at the first time run
    :return 目标项目列表
    """
    # 1. repo category: DBMS
    df_OSDB_github_key_feats = pd.read_csv(dbms_repos_key_feats_path, header='infer', index_col=None, dtype=str)
    if re_preprocess:
        # Complete github_repo_id data with github_repo_link by GitHub API
        df_OSDB_github_key_feats["github_repo_id"] = df_OSDB_github_key_feats.apply(complete_github_repo_id, axis=1)
        df_OSDB_github_key_feats.to_csv(dbms_repos_key_feats_path, header=True, index=False)

        # Complete License_info data by GitHub API
        df_OSDB_github_key_feats["License_info"] = df_OSDB_github_key_feats.apply(complete_license_info, update_license_by_API=False, axis=1)
        df_OSDB_github_key_feats.to_csv(dbms_repos_key_feats_path, header=True, index=False)

    # 2. Has common open source license
    ser_license_updated_validate_common_osl = df_OSDB_github_key_feats.apply(ValidateFunc.check_open_source_license,
                                                                             nan_as_final_false=True,
                                                                             only_common_osl=True, axis=1)
    df_OSDB_github_key_feats = df_OSDB_github_key_feats[ser_license_updated_validate_common_osl]

    # 3. Exist github_repo_id
    ser_has_open_source_github_repo_id = df_OSDB_github_key_feats.apply(ValidateFunc.has_open_source_github_repo_id, axis=1)
    df_OSDB_github_key_feats = df_OSDB_github_key_feats[ser_has_open_source_github_repo_id]

    # 4. Issue related activity threshold
    repo_activity_statistics_dir = os.path.join(os.path.dirname(dbms_repos_key_feats_path), 'repo_activity_statistics')
    repo_i_pr_rec_cnt_path = os.path.join(repo_activity_statistics_dir, 'repo_i_pr_rec_cnt.csv')


    default_table = "opensource.events"
    get_year_constraint = lambda x: f"created_at BETWEEN '{str(x)}-01-01 00:00:00' AND '{str(x + 1)}-01-01 00:00:00'"
    repo_ids = df_OSDB_github_key_feats["github_repo_id"].to_list()
    # repo_id_names_dict = df_OSDB_github_key_feats[["github_repo_id", "github_repo_link"]].to_dict()
    params_condition_dict = {
        "type": "IN ['IssueCommentEvent', 'IssuesEvent', 'PullRequestEvent', 'PullRequestReviewCommentEvent', 'PullRequestReviewEvent']",
    }
    params_condition = get_params_condition(params_condition_dict)
    if re_preprocess:
        conndb = ConnDB()
        conndb.sql = f"""
        SELECT 
            repo_id,
            anyHeavy(repo_name) as repo_name,
            COUNT(*) as i_pr_rec_cnt 
        FROM {default_table}
        WHERE platform = 'GitHub' 
            AND {params_condition}
            AND repo_id IN ('{"','".join(repo_ids)}')
            AND {get_year_constraint(year)}
        GROUP BY repo_id
        ORDER BY i_pr_rec_cnt DESC;"""
        # print(conndb.sql)

        # 从数据库获取项目日志的统计信息
        conndb.execute()
        df_repo_i_pr_rec_cnt = conndb.rs
        df_repo_i_pr_rec_cnt.to_csv(repo_i_pr_rec_cnt_path, header=True, index=False, encoding='utf-8')
    else:
        df_repo_i_pr_rec_cnt = pd.read_csv(repo_i_pr_rec_cnt_path, index_col=None)

    # set issue related activity threshold
    i_pr_rec_cnt_threshold = 10
    df_target_repo = df_repo_i_pr_rec_cnt[df_repo_i_pr_rec_cnt["i_pr_rec_cnt"] >= i_pr_rec_cnt_threshold]
    # 输出选中项目数量
    print(f"Selected {len(df_target_repo)} DBMS projects.")
    return df_target_repo


def retrieve_github_data(repo_names, year=2023):
    """
    从GitHub API获取项目数据
    :repo_names 目标项目的repo_names列表
    :return filenames of retrieved data
    """
    sql_param = {
        "table": "opensource.events",
        "start_end_year": [year, year + 1],
    }
    query_repo_log_each_year_to_csv_dir(repo_names, columns=columns_simple, save_dir=dbms_repos_raw_content_dir,
                                        sql_param=sql_param)
    return


if __name__ == '__main__':
    year = 2023
    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]
    dbms_repos_dedup_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_DEDUP_CONTENT_DIR]
    collaboration_relation_extraction_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_CORE_DIR]

    # 步骤1: 数据收集与预处理
    target_repos = select_target_repos(dbms_repos_key_feats_path, year, re_preprocess=False)

    repo_names = target_repos["repo_names"].to_list()
    retrieve_github_data(repo_names, year)
    filenames = get_filenames_by_repo_names(repo_names, year)
    logger.log(logging.INFO, msg=f"Use {len(filenames)} filenames: {filenames}")

    # Preprocess body content
    process_body_content(raw_content_dir=dbms_repos_raw_content_dir, processed_content_dir=dbms_repos_dedup_content_dir,
                         filenames=filenames)
    df_dbms_repos_dict = read_csvs(dbms_repos_dedup_content_dir, filenames=filenames, index_col=0)

    # 步骤2: 引用关系抽取
    # Get repo_keys
    d_repo_record_length = {k: len(df) for k, df in df_dbms_repos_dict.items()}
    d_repo_record_length_sorted = dict(sorted(d_repo_record_length.items(), key=lambda x: x[1], reverse=True))
    repo_keys = list(d_repo_record_length_sorted.keys())
    df_dbms_repos_dict = {k: df_dbms_repos_dict[k] for k in repo_keys}
    logger.log(logging.INFO, msg=f"Validated {len(repo_keys)} repo_keys sorted by the records count: {d_repo_record_length_sorted}")

    # Collaboration Relation extraction
    collaboration_relation_extraction(repo_keys, df_dbms_repos_dict, collaboration_relation_extraction_dir, update_exists=False,
                                      add_mode_if_exists=True, use_relation_type_list=["EventAction", "Reference"], last_stop_index=-1)

    # 步骤3: 引用耦合网络构建
    df_dbms_repo = df_dbms_repos_dict[repo_keys[0]]
    G_repo = build_collab_net(df_dbms_repo, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
                              default_node_types=['src_entity_type', 'tar_entity_type'], default_edge_type="event_type",
                              init_record_as_edge_attrs=True, use_df_col_as_default_type=True, out_g_type='DG')

    # 步骤4: 网络拓扑特征分析
    analyze_degree_distribution(G_repo)
    calculate_centrality_measures(G_repo)
    detect_community_structure(G_repo)
    compute_clustering_coefficient(G_repo)

    # 步骤5: 描述性指标分析
    analyze_reference_type_distribution()
    calculate_issue_metrics(target_repos)
    study_time_evolution(target_repos)

    # 步骤6: 结果汇总与可视化
    generate_experiment_report()
    visualize_results()
