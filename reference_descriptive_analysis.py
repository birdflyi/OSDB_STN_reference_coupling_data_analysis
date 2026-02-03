#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9
import json
# @Time   : 2026/1/10 15:01
# @Author : 'Lou Zehua'
# @File   : reference_descriptive_analysis.py

import os
import pickle
import sys
import traceback

import networkx as nx
import numpy as np

from script.build_dataset.collaboration_relation_extraction import process_body_content, \
    collaboration_relation_extraction, filenames_exist_filter
from script.build_dataset.repo_filter import get_filenames_by_repo_names
from script.complex_network_analysis.build_network.build_Graph import DG2G
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
from GH_CoRE.working_flow import query_repo_log_each_year_to_csv_dir, read_csvs, get_repo_name_fileformat, \
    get_repo_year_filename

from etc import filePathConf
from script.utils.validate import ValidateFunc, complete_license_info, complete_github_repo_id, complete_repo_created_at
from script.complex_network_analysis.Network_params_analysis import get_graph_feature

setup_logging(base_dir=pkg_rootdir)
logger = logging.getLogger(__name__)


def get_matched_repo_name(row: pd.Series, df_ref: pd.DataFrame, repo_name_colname='github_repo_link',
                          repo_id_colname='github_repo_id', copy_columns=None):
    new_row_index = list(row.index) + ["repo_name"]
    if copy_columns is not None:
        copy_columns = list(copy_columns)
        new_row_index += copy_columns
    new_row = row.reindex(new_row_index, fill_value = None)
    latest_repo_id = str(row["repo_id"])
    repo_name_used = row["repo_name_used"]
    for index_ref, row_ref in df_ref.iterrows():
        if repo_name_used == row_ref[repo_name_colname] and latest_repo_id == row_ref[repo_id_colname]:
            new_row["repo_name"] = row_ref[repo_name_colname]
            if copy_columns is not None:
                new_row.update(row_ref[copy_columns])
            break
        else:
            continue
    return new_row


# 步骤2: 数据收集与预处理
def select_target_repos(dbms_repos_key_feats_path, year=2023, re_preprocess=False, i_pr_rec_cnt_threshold=10, ret='name_list'):
    """
    :dbms_repos_key_feats_path 从dbdb.io与dbengines收录列表中选择符合条件的DBMS项目
    :re_preprocess set True at the first time run
    :return 目标项目列表
    """
    # 1. repo category: DBMS
    df_OSDB_github_key_feats = pd.read_csv(dbms_repos_key_feats_path, header='infer', index_col=None, dtype=str)
    if re_preprocess:
        # Complete github_repo_id data with github_repo_link by GitHub API
        df_OSDB_github_key_feats["github_repo_id"] = df_OSDB_github_key_feats.apply(complete_github_repo_id, update_repo_id_by_API=False, axis=1)
        # Complete License_info data by GitHub API
        df_OSDB_github_key_feats["License_info"] = df_OSDB_github_key_feats.apply(complete_license_info, update_license_by_API=False, axis=1)
        # Complete License_info data by GitHub API
        df_OSDB_github_key_feats["repo_created_at"] = df_OSDB_github_key_feats.apply(complete_repo_created_at, update_repo_created_at_by_API=False, axis=1)
        df_OSDB_github_key_feats.to_csv(dbms_repos_key_feats_path, header=True, index=False)

    # 2. Has common open source license
    ser_license_updated_validate_common_osl = df_OSDB_github_key_feats.apply(ValidateFunc.check_open_source_license,
                                                                             nan_as_final_false=True,
                                                                             only_common_osl=True, axis=1)
    df_OSDB_github_key_feats = df_OSDB_github_key_feats[ser_license_updated_validate_common_osl]

    # 3. Exist github_repo_id
    ser_has_open_source_github_repo_id = df_OSDB_github_key_feats.apply(ValidateFunc.has_open_source_github_repo_id, axis=1)
    df_OSDB_github_key_feats = df_OSDB_github_key_feats[ser_has_open_source_github_repo_id]

    # 4. Get Issue related activity
    repo_activity_statistics_dir = os.path.join(os.path.dirname(dbms_repos_key_feats_path), 'repo_activity_statistics')
    repo_i_pr_rec_cnt_path = os.path.join(repo_activity_statistics_dir, 'repo_i_pr_rec_cnt.csv')


    default_table = "opensource.events"
    get_year_constraint = lambda x: f"created_at BETWEEN '{str(x)}-01-01 00:00:00' AND '{str(x + 1)}-01-01 00:00:00'"
    repo_ids = df_OSDB_github_key_feats["github_repo_id"].to_list()
    params_condition_dict = {
        "type": "IN ['IssueCommentEvent', 'IssuesEvent', 'PullRequestEvent', 'PullRequestReviewCommentEvent', 'PullRequestReviewEvent']",
    }
    params_condition = get_params_condition(params_condition_dict)
    if re_preprocess:
        conndb = ConnDB()
        conndb.sql = f"""
        SELECT 
            repo_id,
            repo_name as repo_name_used,
            COUNT(*) as i_pr_rec_cnt 
        FROM {default_table}
        WHERE platform = 'GitHub' 
            AND {params_condition}
            AND repo_id IN ('{"','".join(repo_ids)}')
            AND {get_year_constraint(year)}
        GROUP BY repo_id, repo_name
        ORDER BY i_pr_rec_cnt DESC;"""
        # print(conndb.sql)
        # 从数据库获取项目日志的统计信息
        conndb.execute()
        df_repo_i_pr_rec_cnt = conndb.rs  # columns: ["repo_id", "repo_name_used", "i_pr_rec_cnt"]
        # 由于存在仓库更名，不同的repo_name可以对应相同的repo_id，不能直接对repo_id set_index。
        copy_columns = ["repo_created_at"]
        df_repo_i_pr_rec_cnt = df_repo_i_pr_rec_cnt.apply(get_matched_repo_name, df_ref=df_OSDB_github_key_feats, copy_columns=copy_columns, axis=1, result_type='expand')
        df_repo_i_pr_rec_cnt = df_repo_i_pr_rec_cnt[["repo_id", "repo_name", "repo_name_used", "i_pr_rec_cnt"] + copy_columns]
        df_repo_i_pr_rec_cnt.to_csv(repo_i_pr_rec_cnt_path, header=True, index=False, encoding='utf-8')
    else:
        df_repo_i_pr_rec_cnt = pd.read_csv(repo_i_pr_rec_cnt_path, index_col=None)

    # The repo_name exists in github event log of year=2023 and is included in the 'github_repo_link' field
    #   of dbms_repos_key_feats_path
    df_repo_i_pr_rec_cnt = df_repo_i_pr_rec_cnt[df_repo_i_pr_rec_cnt["repo_name"].notna()]

    # 5. Issue or PullRequest record count filter
    df_target_repos = df_repo_i_pr_rec_cnt[df_repo_i_pr_rec_cnt["i_pr_rec_cnt"] >= i_pr_rec_cnt_threshold]
    if ret == 'name_list':
        target_repos = df_target_repos["repo_name"].to_list()
    elif ret == 'dataframe':
        target_repos = df_target_repos
        target_repos["repo_id"] = target_repos["repo_id"].astype(str)
    else:
        raise ValueError("ret must be 'name_list' or 'dataframe'!")
    return target_repos


def retrieve_github_data(repo_names, year=2023, use_new_ck_data=False):
    """
    从GitHub API获取项目数据
    :repo_names 目标项目的repo_names列表
    :return filenames of retrieved data
    """
    sql_param = {
        "table": "opensource.events",
        "start_end_year": [year, year + 1],
    }
    if use_new_ck_data:
        # 最新的clickhouse不再支持以下字段：
        deactivated_fields = ['id', 'repo_description', 'delete_ref', 'delete_ref_type', 'create_ref', 'create_ref_type', 'create_master_branch', 'member_id', 'member_login']
        columns = [field for field in columns_simple if field not in deactivated_fields]
    else:
        columns = columns_simple
    query_repo_log_each_year_to_csv_dir(repo_names, columns=columns, save_dir=dbms_repos_raw_content_dir,
                                        sql_param=sql_param)
    return


def dedup_x_y_keep_na_by_z(df, subset=None, keep='first'):
    """
    对DataFrame进行去重处理：
    - 当z列相同时，对x,y相同的行进行去重
    - x,y不为空的行才参与去重
    - y可能为None，此时不作去重并保留
    - 保持原始行次序
    """
    if subset is None:
        subset = ['x', 'y', 'z']
    colname_x, colname_y, colname_z = tuple(subset)
    # 创建一个副本，避免修改原数据
    df_copy = df.copy()

    # 标记x,y都不为空的行
    mask = (df_copy[colname_x].notna()) & (df_copy[colname_y].notna())

    # 对x,y都不为空的行，按z分组，保留第一次出现的行
    if mask.any():
        # 为所有行添加临时索引以保持原始顺序
        df_copy['original_index'] = df_copy.index

        # 对满足条件的行进行去重
        deduped = df_copy[mask].drop_duplicates(subset=subset, keep=keep)

        # 保留不满足条件的行（y为None的行）
        non_dup_rows = df_copy[~mask]

        # 合并结果，保持原始顺序
        df_result = pd.concat([deduped, non_dup_rows], ignore_index=True)

        # 按临时索引排序以保持原始顺序
        df_result = df_result.sort_values('original_index').set_index('original_index')
        df_result.index.name = None
    else:
        # 如果没有x,y都不为空的行，直接返回原数据
        df_result = df_copy
    return df_result


def is_reponame_repokey_matched(repo_name: str, repo_key: str):
    match_flag = False
    repo_name_fileformat = get_repo_name_fileformat(repo_name)
    filename = get_repo_year_filename(repo_name_fileformat, year)
    if filename == repo_key + '.csv':
        match_flag = True
    return match_flag


def granu_agg(row: pd.Series, repo_id=None):
    if row["src_entity_type"] == "Actor":
        row["src_entity_id_agg"] = row["src_entity_id"]
        row["src_entity_type_agg"] = row["src_entity_type"]
    else:
        row["src_entity_id_agg"] = "R_" + str(repo_id)
        row["src_entity_type_agg"] = "Repo"

    tar_entity_id_agg = None
    tar_entity_type_agg = "Object"
    tar_entity_objnt_prop_dict = row["tar_entity_objnt_prop_dict"]
    try:
        if np.isnan(float(tar_entity_objnt_prop_dict)):
            tar_entity_objnt_prop_dict = None
    except:
        pass

    if tar_entity_objnt_prop_dict is None:  # all of GitHub_Other_Service, GitHub_Service_External_Links
        pass
    else:
        try:
            tar_entity_objnt_prop_dict = dict(tar_entity_objnt_prop_dict)
        except Exception:
            prop_str = str(tar_entity_objnt_prop_dict)
            try:
                tar_entity_objnt_prop_dict = json.loads(prop_str)
            except json.JSONDecodeError:
                # Swap the two quotation marks and try to parse again
                prop_str = prop_str.replace('"', '$').replace("'", '"').replace('$', "'")
                try:
                    tar_entity_objnt_prop_dict = json.loads(prop_str)
                except json.JSONDecodeError:
                    try:
                        tar_entity_objnt_prop_dict = dict(eval(prop_str))
                    except Exception:
                        prop_str = prop_str.replace("'", '"')  # Forced analysis with [\', \"] mixed mode
                        tar_entity_objnt_prop_dict = json.loads(prop_str)
        if "repo_id" in tar_entity_objnt_prop_dict.keys():
            if tar_entity_objnt_prop_dict["repo_id"] is not None:  # Except for unknown sha like fragment
                tar_entity_id_agg = "R_" + str(tar_entity_objnt_prop_dict["repo_id"])
                tar_entity_type_agg = "Repo"
        elif "actor_id" in tar_entity_objnt_prop_dict.keys():
            if tar_entity_objnt_prop_dict["actor_id"] is not None:
                tar_entity_id_agg = "A_" + str(tar_entity_objnt_prop_dict["actor_id"])
                tar_entity_type_agg = "Actor"
        else:
            pass  # can not parse
    row["tar_entity_id_agg"] = tar_entity_id_agg
    row["tar_entity_type_agg"] = tar_entity_type_agg
    return row


def set_entity_type_fine_grained(row: pd.Series):
    ent_type = "GitHub_Service_External_Links"
    if row["tar_entity_type_agg"] == "Object":  # GitHub_Other_Service and GitHub_Service_External_Links and other wrong pattern has no id
        if row["tar_entity_match_pattern_type"] in ["GitHub_Other_Service", "GitHub_Service_External_Links"]:
            ent_type = row["tar_entity_match_pattern_type"]
        else:
            pass  # Can not get a valid node response from GitHub REST API or GitHub GraphQL. Regard as GitHub_Service_External_Links.
    else:  # row["tar_entity_type"] have Fine grained type when row["tar_entity_type"] != "Object", especially for Issue_PR and SHA pattern
        if row["tar_entity_type"] == "Object":
            ent_type = row["tar_entity_match_pattern_type"]
        else:
            ent_type = row["tar_entity_type"]  # for Issue, IssueComment, PullRequest, PullRequestReviewComment and Commit
    row["tar_entity_type_fine_grained"] = ent_type
    return row


def write_gexf_with_forced_types(G, filepath, forced_types=None, repl_None_str=""):
    """
    强制指定GEXF文件中属性的类型，避免NetworkX自动推断错误。
    适用于如 'tar_entity_id'、'src_entity_id' 等本应为字符串却因存在None被误判为 double 的场景。

    :param G: NetworkX 图对象
    :param filepath: 输出文件路径
    :param forced_types: 字典，格式为 {属性名: 类型字符串}，如 {'tar_entity_id': 'string'}
    """
    if forced_types is None:
        forced_types = {}

    # 为图添加自定义元数据，NetworkX 会读取此信息来覆盖自动推断
    G.graph['_attribute_types'] = forced_types

    # 确保所有指定属性的值都是字符串（避免数值型干扰）
    for _, _, edge_data in G.edges(data=True):
        for attr_name in forced_types:
            if attr_name in edge_data:
                edge_data[attr_name] = str(edge_data[attr_name]) if edge_data[attr_name] is not None else repl_None_str

    # 保存GEXF文件
    nx.write_gexf(G, filepath)
    return G


if __name__ == '__main__':
    year = 2023
    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]
    dbms_repos_dedup_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_DEDUP_CONTENT_DIR]
    dbms_repos_gh_core_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_GH_CORE_DIR]
    dbms_repos_gh_core_ref_node_agg_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_GH_CORE_REF_NODE_AGG_DIR]
    graph_network_dir = filePathConf.absPathDict[filePathConf.GRAPH_NETWORK_DIR]
    flag_skip_existing_files = True
    ref_dedup_by_event_id = False

    # 步骤1: 数据收集与预处理
    logger.info(f"-------------Step 1. Data preprocessing-------------")
    df_target_repos = select_target_repos(dbms_repos_key_feats_path, year, re_preprocess=False, ret="dataframe")
    repo_names = df_target_repos["repo_name"].to_list()
    logger.info(f"Selected {len(repo_names)} DBMS projects.")

    filenames = get_filenames_by_repo_names(repo_names, year)
    if flag_skip_existing_files:  # Use the downloaded DBMS sample dataset
        filenames = [f_name for f_name in filenames if os.path.isfile(os.path.join(dbms_repos_raw_content_dir, f_name))]
        logger.info(f"Missing {len(repo_names) - len(filenames)} DBMS projects in old sample data. "
                    f"Repo num loss ratio: {round((1-len(filenames)/len(repo_names))*100,2) if len(repo_names) else 0}%")
    else:
        retrieve_github_data(repo_names, year)
    logger.info(f"Use {len(filenames)} filenames: {filenames}")

    # Preprocess body content
    process_body_content(raw_content_dir=dbms_repos_raw_content_dir, processed_content_dir=dbms_repos_dedup_content_dir,
                         filenames=filenames, dedup_content_overwrite_all=not flag_skip_existing_files)

    # 步骤2: 引用关系抽取
    logger.info(f"------------Step 2. Extract Relationship------------")
    keep_part = 'is_not_file' if flag_skip_existing_files else 'all'
    filenames_need_extract = filenames_exist_filter(dbms_repos_dedup_content_dir, filenames, keep_part=keep_part)
    # Get repo_keys
    logger.info(f"Read data from {dbms_repos_dedup_content_dir}. This may take a lot of time...")
    df_dbms_repos_dict = read_csvs(dbms_repos_dedup_content_dir, filenames=filenames_need_extract, index_col=0)
    logger.info(f"Read completed.")

    d_repo_record_length = {k: len(df) for k, df in df_dbms_repos_dict.items()}
    d_repo_record_length_sorted = dict(sorted(d_repo_record_length.items(), key=lambda x: x[1], reverse=True))
    repo_keys_need_extract = list(d_repo_record_length_sorted.keys())
    df_dbms_repos_dict = {k: df_dbms_repos_dict[k] for k in repo_keys_need_extract}
    logger.info(f"The {len(d_repo_record_length_sorted)} repo_keys to be processed sorted by the records count: {d_repo_record_length_sorted}")

    # Collaboration relation extraction
    logger.info(f"Collaboration relation extraction start...")
    collaboration_relation_extraction(repo_keys_need_extract, df_dbms_repos_dict, dbms_repos_gh_core_dir,
                                      update_exists=not flag_skip_existing_files, add_mode_if_exists=True,
                                      use_relation_type_list=["EventAction", "Reference"], last_stop_index=-1)
    logger.info(f"Collaboration relation extraction completed.")

    # 步骤3: 引用耦合网络构建
    logger.info(f"-----------Step 3. Build Reference Network-----------")
    # 边ref去重、结点repo actor粒度聚合
    if not flag_skip_existing_files:
        # read relations
        df_dbms_repos_dict = read_csvs(dbms_repos_gh_core_dir, filenames=filenames, index_col=None)
        # reference filter
        df_dbms_repos_ref_dict = {k: df[df["relation_type"] == "Reference"] for k, df in df_dbms_repos_dict.items()}
        # deduplicate reference relation if ref_dedup_by_event_id = True: deduplicate the same notna <src_entity_id, tar_entity_id> pairs with a same event_id
        if ref_dedup_by_event_id:
            df_dbms_repos_ref_dict = {k: dedup_x_y_keep_na_by_z(df, subset=['src_entity_id', 'tar_entity_id', 'event_id'], keep='first') for k, df in df_dbms_repos_ref_dict.items()}
        # granularity aggregation
        df_dbms_repos_ref_node_agg_dict = {}
        for repo_key, df_dbms_repo_ref in list(df_dbms_repos_ref_dict.items()):
            # get repo_id by repo_key using df_target_repos
            repo_id_match_flags = df_target_repos.apply(lambda row: row['repo_id'] if is_reponame_repokey_matched(row['repo_name'], repo_key) else None, axis=1)
            repo_id = repo_id_match_flags.dropna().iloc[0] if not repo_id_match_flags.dropna().empty else None
            df_dbms_repo_ref_node_agg = df_dbms_repo_ref.apply(granu_agg, axis=1, repo_id=repo_id)  # repo_id as source repo id
            df_dbms_repo_ref_node_agg = df_dbms_repo_ref_node_agg.apply(set_entity_type_fine_grained, axis=1)
            temp_save_path = os.path.join(dbms_repos_gh_core_ref_node_agg_dir, f'{repo_key}.csv')
            df_dbms_repo_ref_node_agg.to_csv(temp_save_path, header=True, index=False, encoding='utf-8')
            df_dbms_repos_ref_node_agg_dict[repo_key] = df_dbms_repo_ref_node_agg
    else:
        pass
    logger.info("Node granularity has been aggregated.")

    # build reference network
    logger.info("Build reference network...")
    use_repo_nodes_only = True
    dg_dbms_repos_ref_net_node_agg_filename = "dg_dbms_repos_ref_net_node_agg.gexf"
    homo_dg_dbms_repos_ref_net_node_agg_filename = "homo_dg_dbms_repos_ref_net_node_agg.gexf"
    dg_dbms_repos_ref_net_node_agg_path = os.path.join(graph_network_dir, dg_dbms_repos_ref_net_node_agg_filename)
    homo_dg_dbms_repos_ref_net_node_agg_path = os.path.join(graph_network_dir, homo_dg_dbms_repos_ref_net_node_agg_filename)
    forced_types = {'src_entity_id': 'string', 'tar_entity_id': 'string'}
    if not flag_skip_existing_files:
        # read aggregated relations
        df_dbms_repos_ref_node_agg_dict = read_csvs(dbms_repos_gh_core_ref_node_agg_dir, filenames=filenames, index_col=None)
        # descent order by records length
        temp_repoKey_recLen_dict = {k: len(df) for k, df in df_dbms_repos_ref_node_agg_dict.items()}
        temp_repoKey_recLen_sorted_dict = dict(sorted(temp_repoKey_recLen_dict.items(), key=lambda x: x[1], reverse=True))
        repo_keys = list(temp_repoKey_recLen_sorted_dict.keys())
        df_dbms_repos_ref_node_agg_dict = {k: df_dbms_repos_ref_node_agg_dict[k] for k in repo_keys}
        # Merge the graph_network of multiple repos
        base_graph = nx.MultiDiGraph()
        G_repo = base_graph
        for repo_key, df_dbms_repo in list(df_dbms_repos_ref_node_agg_dict.items()):
            if df_dbms_repo is not None:
                if len(df_dbms_repo):
                    df_dbms_repo = df_dbms_repo.dropna(subset=['src_entity_id_agg', 'tar_entity_id_agg'], how='any')
                    # build graph_network
                    G_repo = build_collab_net(df_dbms_repo, src_tar_colnames=['src_entity_id_agg', 'tar_entity_id_agg'], base_graph=base_graph,
                                              default_node_types=['src_entity_type_agg', 'tar_entity_type_agg'], default_edge_type="event_type",
                                              init_record_as_edge_attrs=True, use_df_col_as_default_type=True, out_g_type='DG')
                    base_graph = G_repo
        write_gexf_with_forced_types(G_repo, dg_dbms_repos_ref_net_node_agg_path, forced_types=forced_types)
        logger.info(f"{dg_dbms_repos_ref_net_node_agg_path} saved!")
        if use_repo_nodes_only:
            nodes_to_remove = [n for n, data in G_repo.nodes(data=True) if data.get('node_type') != 'Repo']
            G_repo.remove_nodes_from(nodes_to_remove)
            write_gexf_with_forced_types(G_repo, homo_dg_dbms_repos_ref_net_node_agg_path, forced_types=forced_types)
            logger.info(f"{homo_dg_dbms_repos_ref_net_node_agg_path} saved!")
    else:
        if use_repo_nodes_only and os.path.exists(homo_dg_dbms_repos_ref_net_node_agg_path):
            G_repo = nx.read_gexf(homo_dg_dbms_repos_ref_net_node_agg_path)
            logger.info(f"{homo_dg_dbms_repos_ref_net_node_agg_path} already exists!")
        else:
            G_repo = nx.read_gexf(dg_dbms_repos_ref_net_node_agg_path)
            logger.info(f"Read {homo_dg_dbms_repos_ref_net_node_agg_path} ...")
            if use_repo_nodes_only:
                nodes_to_remove = [n for n, data in G_repo.nodes(data=True) if data.get('node_type') != 'Repo']
                G_repo.remove_nodes_from(nodes_to_remove)
                write_gexf_with_forced_types(G_repo, homo_dg_dbms_repos_ref_net_node_agg_path, forced_types=forced_types)
                logger.info(f"{homo_dg_dbms_repos_ref_net_node_agg_path} saved!")

    G_repo_ud = DG2G(G_repo, only_upper_triangle=False, multiplicity=True, double_self_loop=True)
    feat = ["len_nodes", "len_edges", "edge_density", "is_sparse", "avg_deg", "avg_clustering",
            "lcc_node_coverage_ratio", "lcc_len_nodes", "lcc_len_edges", "lcc_edge_density", "lcc_diameter",
            "lcc_assort_coe", "lcc_avg_dist"]
    # graph_feature_record_complex_network = get_graph_feature(G_repo_ud, feat=feat)
    # df_dbms_repos_ref_net_node_agg_feat = pd.DataFrame.from_dict(graph_feature_record_complex_network, orient='index')
    # df_dbms_repos_ref_net_node_agg_feat_path = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], 'analysis_results/homo_dbms_repos_ref_net_node_agg_feat.csv')
    # df_dbms_repos_ref_net_node_agg_feat.to_csv(df_dbms_repos_ref_net_node_agg_feat_path, header=False, index=True)
    # logger.info(f"{df_dbms_repos_ref_net_node_agg_feat_path} saved!")

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

    G_repo_ud = DG2G(G_repo, only_upper_triangle=False, multiplicity=True, double_self_loop=True)
    graph_feature_record_complex_network = get_graph_feature(G_repo_ud, feat=feat)
    df_dbms_repos_ref_net_node_agg_feat = pd.DataFrame.from_dict(graph_feature_record_complex_network, orient='index')
    feat_filename = f"homo{'_only' if only_dbms_repo else ''}_dbms_repos_ref_net_node_agg{'_dsl' if drop_self_loop else ''}_feat.csv"
    df_dbms_repos_ref_net_node_agg_feat_path = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], f'analysis_results/{feat_filename}')
    df_dbms_repos_ref_net_node_agg_feat.to_csv(df_dbms_repos_ref_net_node_agg_feat_path, header=False, index=True)
    logger.info(f"{df_dbms_repos_ref_net_node_agg_feat_path} saved!")

    # # 步骤5: 描述性指标分析
    # analyze_reference_type_distribution(dbms_repos_gh_core_ref_node_agg_dir, filenames=filenames)
    # calculate_issue_metrics(target_repos)
    # study_time_evolution(target_repos)

    # # 步骤4: 网络拓扑特征分析
    # analyze_degree_distribution(G_repo)
    # calculate_centrality_measures(G_repo)
    # detect_community_structure(G_repo)
    # compute_clustering_coefficient(G_repo)
    #
    # # 步骤6: 结果汇总与可视化
    # generate_experiment_report()
    # visualize_results()
