#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2026/1/10 15:01
# @Author : 'Lou Zehua'
# @File   : reference_descriptive_analysis.py

import os
import sys
import time

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

import json
import logging
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import traceback

from collections import Counter
from scipy import stats, optimize

from GH_CoRE.data_dict_settings import columns_simple, body_columns_dict, event_columns_dict, re_ref_patterns
from GH_CoRE.model import ObjEntity, Attribute_getter, Entity_search
from GH_CoRE.utils import get_params_condition
from GH_CoRE.utils.conndb import ConnDB
from GH_CoRE.utils.logUtils import setup_logging
from GH_CoRE.working_flow import query_repo_log_each_year_to_csv_dir, read_csvs, get_repo_name_fileformat, \
    get_repo_year_filename

from etc import filePathConf
from script.build_dataset.collaboration_relation_extraction import process_body_content, \
    collaboration_relation_extraction, filenames_exist_filter
from script.build_dataset.repo_filter import get_filenames_by_repo_names
from script.complex_network_analysis.build_network.build_Graph import DG2G
from script.complex_network_analysis.build_network.build_gh_collab_net import build_collab_net
from script.complex_network_analysis.Network_params_analysis import get_graph_feature
from script.utils.validate import ValidateFunc, complete_license_info, complete_github_repo_id, complete_repo_created_at

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


def integrate_category_label(row: pd.Series):
    mix_src_sep = "#dbdbio>|<dbengines#"
    category_label_str = str(row["category_label"]).replace(mix_src_sep, ",") if pd.notna(row["category_label"]) else ""
    category_labels = list(set(category_label_str.split(",")))
    category_labels_not_empty = [e for e in category_labels if e]
    row["category_label"] = ','.join(category_labels_not_empty) if len(category_labels_not_empty) else ""
    return row


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
        copy_columns = ["repo_created_at", "category_label"]
        df_repo_i_pr_rec_cnt = df_repo_i_pr_rec_cnt.apply(get_matched_repo_name, df_ref=df_OSDB_github_key_feats, copy_columns=copy_columns, axis=1, result_type='expand')
        df_repo_i_pr_rec_cnt = df_repo_i_pr_rec_cnt.apply(integrate_category_label, axis=1)
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


def is_reponame_repokey_matched(repo_name: str, repo_key: str, year=2023):
    match_flag = False
    repo_name_fileformat = get_repo_name_fileformat(repo_name)
    filename = get_repo_year_filename(repo_name_fileformat, year)
    if filename == repo_key + '.csv':
        match_flag = True
    return match_flag


def get_repo_id_by_repo_key(repo_key, df_repo_i_pr_rec_cnt, year=2023):
    repo_id_match_flags = df_repo_i_pr_rec_cnt.apply(
        lambda row: row['repo_id'] if is_reponame_repokey_matched(row['repo_name'], repo_key, year) else None, axis=1)
    repo_id = repo_id_match_flags.dropna().iloc[0] if not repo_id_match_flags.dropna().empty else None
    return repo_id


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
            if ent_type == "Issue_PR":
                try:
                    tar_entity_objnt_prop_dict = eval(row["tar_entity_objnt_prop_dict"])
                except:
                    tar_entity_objnt_prop_dict = {}
                if isinstance(tar_entity_objnt_prop_dict, dict):
                    repo_id = tar_entity_objnt_prop_dict.get("repo_id")
                    issue_number = tar_entity_objnt_prop_dict.get("issue_number")
                    if repo_id and issue_number:
                        row["tar_entity_type"] = Attribute_getter.__get_issue_type(repo_id, issue_number)
                        ent_type = row["tar_entity_type"]
                        tar_entity = ObjEntity(ent_type)
                        tar_entity.set_val(tar_entity_objnt_prop_dict)
                        row["tar_entity_id"] = tar_entity.__repr__(brief=True) if tar_entity.__PK__ else None
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


# 按列值分割dataframe
def split_dataframe_by_column(df, column_name):
    # 用于存储结果的字典
    result_dict = {}

    # 遍历每一行
    for index, row in df.iterrows():
        # 获取该行的值并分割成集合
        values = set(row[column_name].split(','))

        # 为每个值创建一个子df
        for value in values:
            # 如果该值不在结果字典中，创建新的df
            if value not in result_dict:
                result_dict[value] = df[df[column_name].str.contains(value, na=False)].copy()
            else:
                # 否则，将当前行添加到已存在的df中
                temp_df = df[df[column_name].str.contains(value, na=False)].copy()
                result_dict[value] = pd.concat([result_dict[value], temp_df], ignore_index=True).drop_duplicates()

    return result_dict


def analyze_referenced_type_distribution(df_dbms_repos_ref_node_agg_dict, df_repo_i_pr_rec_cnt=None,
                                         src_type_col="src_entity_type",
                                         tar_type_col="tar_entity_type_fine_grained"):
    logger.info("a. analyze_referenced_type_distribution...")
    dbms_repo_num = len(df_dbms_repos_ref_node_agg_dict)
    logger.info(f"DMBS repo number: {dbms_repo_num}.")
    unique_repo_ids = set()
    unique_actor_ids = set()
    src_ent_id_col = "src_entity_id_agg"
    tar_ent_id_col = "tar_entity_id_agg"
    src_tar_id_cols = [src_ent_id_col, tar_ent_id_col]
    for repo_key, df_ref in df_dbms_repos_ref_node_agg_dict.items():
        for col in src_tar_id_cols:
            repo_id_values = df_ref[df_ref[col].astype(str).str.startswith('R_')][col]
            unique_repo_ids.update(repo_id_values)
            actor_id_values = df_ref[df_ref[col].astype(str).str.startswith('A_')][col]
            unique_actor_ids.update(actor_id_values)
    Repo_num = len(unique_repo_ids)
    Actor_num = len(unique_actor_ids)
    logger.info(f"Repo number in reference network: {Repo_num}.")
    logger.info(f"Actor number in reference network: {Actor_num}.")

    github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
    total_counter = Counter()
    total_count = 0
    for repo_key, df_ref in df_dbms_repos_ref_node_agg_dict.items():
        valid_types = df_ref[src_type_col].dropna()
        if len(valid_types) > 0:
            referencing_entity_type_counts = df_ref[src_type_col].value_counts()
            total_counter.update(referencing_entity_type_counts.to_dict())
            total_count += len(df_ref)
    logger.info(f"Total count of referencing records: {total_count}.")
    df_referencing_type_distribution = pd.DataFrame({
        'referencing_entity_type': list(total_counter.keys()),
        'count': list(total_counter.values()),
        'proportion': [count / total_count for count in total_counter.values()]
    })
    df_referencing_type_distribution = df_referencing_type_distribution.sort_values('count', ascending=False).reset_index(drop=True)
    df_referencing_type_distribution.to_csv(os.path.join(github_osdb_data_dir, f"analysis_results/ref_type_dist/df_referencing_type_distribution_len_{total_count}.csv"), header=True, index=False, encoding='utf-8')

    total_counter = Counter()
    total_count = 0
    for repo_key, df_ref in df_dbms_repos_ref_node_agg_dict.items():
        valid_types = df_ref[tar_type_col].dropna()
        if len(valid_types) > 0:
            referenced_entity_type_counts = df_ref[tar_type_col].value_counts()
            total_counter.update(referenced_entity_type_counts.to_dict())
            total_count += len(df_ref)
    logger.info(f"Total count of referenced records: {total_count}.")
    df_referenced_type_distribution = pd.DataFrame({
        'referenced_entity_type': list(total_counter.keys()),
        'count': list(total_counter.values()),
        'proportion': [count / total_count for count in total_counter.values()]
    })
    df_referenced_type_distribution = df_referenced_type_distribution.sort_values('count', ascending=False).reset_index(drop=True)
    github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
    df_referenced_type_distribution.to_csv(os.path.join(github_osdb_data_dir, f"analysis_results/ref_type_dist/df_referenced_type_distribution_len_{total_count}.csv"), header=True, index=False, encoding='utf-8')

    # 计算自引率
    self_ref_ratio_dicts = []
    if df_repo_i_pr_rec_cnt is None:
        repo_i_pr_rec_cnt_path = os.path.join(github_osdb_data_dir, 'repo_activity_statistics/repo_i_pr_rec_cnt.csv')
        df_repo_i_pr_rec_cnt = pd.read_csv(repo_i_pr_rec_cnt_path, index_col=None)
    for repo_key, df_ref in df_dbms_repos_ref_node_agg_dict.items():
        non_null_mask = df_ref[src_ent_id_col].notna() | df_ref[tar_ent_id_col].notna()
        df_filtered = df_ref[non_null_mask]

        # 在过滤后的数据中，计算src列和tar列字符串值完全相同的行数
        if len(df_filtered) == 0:
            self_ref_ratio = 0.0
        else:
            match_count = (df_filtered[src_ent_id_col] == df_filtered[tar_ent_id_col]).sum()
            self_ref_ratio = match_count / len(df_filtered)
        repo_id = str(list(df_ref[src_ent_id_col])[0]).lstrip("R_")  if len(df_ref) else None
        repo_names_matched = df_repo_i_pr_rec_cnt[df_repo_i_pr_rec_cnt["repo_id"].astype(str) == str(repo_id)]
        repo_name = repo_names_matched['repo_name'].dropna().iloc[0] if not repo_names_matched['repo_name'].dropna().empty else None
        self_ref_ratio_dicts.append({'repo_id': repo_id, 'repo_name': repo_name, 'self_ref_ratio': self_ref_ratio})
    df_self_ref_ratio = pd.DataFrame(self_ref_ratio_dicts)
    df_self_ref_ratio.to_csv(os.path.join(github_osdb_data_dir, f"analysis_results/ref_type_dist/df_self_ref_ratio.csv"), header=True, index=False, encoding='utf-8')
    logger.info(f"Sef reference ratio describe: \n{df_self_ref_ratio['self_ref_ratio'].describe()}.")
    logger.info("Analyzed reference type distribution successfully.")
    return


issue_related_entity_types = ["Issue", "IssueComment", "PullRequest", "PullRequestReview", "PullRequestReviewComment"]
comment_related_entity_types = ["IssueComment", "PullRequestReview", "PullRequestReviewComment"]


def get_entity_type_abbr(entity_type):
    entity_type_abbr = None
    entity_type_dict = ObjEntity.E.get(str(entity_type), None)
    if isinstance(entity_type_dict, dict):
        entity_type_abbr = entity_type_dict.get('ABBR')
    return entity_type_abbr


def is_entity_related(src_entity_id: str, related_entity_types=None):
    flag = False
    if related_entity_types is None:
        related_entity_types = issue_related_entity_types
    for entity_type in related_entity_types:
        entity_type_abbr = get_entity_type_abbr(entity_type)
        if entity_type_abbr:
            prefix = entity_type_abbr + "_"
            if src_entity_id.startswith(prefix):
                flag = True
                break
    return flag


def extract_repo_issue_mixed_id(src_entity_id):
    # 匹配模式：_ 后面跟着数字#数字 的格式 e.g. input: "PRRC_346976717#3401#r1281755060"; return: "346976717#3401"
    if not src_entity_id:
        return None
    match = re.search(r'_(\d+#\d+)', src_entity_id)
    if match:
        return match.group(1)
    return None


def calculate_issue_referencing_metrics(df_dbms_repos_ref_dict, df_repo_i_pr_rec_cnt=None, strict_comment_type=False):
    """
    计算活跃议题数、新增议评率、新增评引率分布
    活跃议题数（活跃Issue和pr数）
    新增议评率（含Issue和PR body的新增Comment事件数/活跃Issue和pr数）
    新增评引率（Comment事件中新增引用数/含Issue和PR body的新增Comment事件数）
    """
    logger.info("b. calculate_issue_referencing_metrics...")
    if df_repo_i_pr_rec_cnt is None:
        github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
        repo_i_pr_rec_cnt_path = os.path.join(github_osdb_data_dir, 'repo_activity_statistics/repo_i_pr_rec_cnt.csv')
        df_repo_i_pr_rec_cnt = pd.read_csv(repo_i_pr_rec_cnt_path, index_col=None)
    # 用于存储每个项目的指标
    issue_referencing_metrics_list = []
    for repo_key, df_ref in df_dbms_repos_ref_dict.items():
        # 统计不同类型的实体数量
        # 有事件活跃的Issue和PullRequest数量 (引用源为Issue的issue_id数)
        df_ref_issue_related = df_ref[
            df_ref["src_entity_id"].apply(is_entity_related, related_entity_types=issue_related_entity_types)]
        ser_repo_issue_mixed_ids = df_ref_issue_related["src_entity_id"].apply(extract_repo_issue_mixed_id)
        issue_pr_active_id_count = len(ser_repo_issue_mixed_ids.dropna().unique())

        # 新增Comment事件数
        if strict_comment_type:  # 不含Issue和PR body
            df_ref_comment_related = df_ref[
                df_ref["src_entity_id"].apply(is_entity_related, related_entity_types=comment_related_entity_types)]
        else:  # 含Issue和PR body
            df_ref_comment_related = df_ref_issue_related
        ser_comment_related_ids = df_ref_comment_related["src_entity_id"]
        comment_related_id_count = len(ser_comment_related_ids.dropna().unique())

        # Comment事件中新增引用数
        # 不含Issue和PR相关事件以外的引用
        comment_body_ref_count = len(df_ref_comment_related)

        # 项目新增引用数
        total_ref_count = len(df_ref)

        # 计算指标
        active_issue_num = issue_pr_active_id_count

        if issue_pr_active_id_count > 0:
            comment_per_issue = comment_related_id_count / issue_pr_active_id_count
        else:
            comment_per_issue = 0

        if comment_related_id_count > 0:
            referencing_per_comment = comment_body_ref_count / comment_related_id_count
        else:
            referencing_per_comment = 0

        repo_id = None
        ser_src_entity_id = df_ref[(~df_ref["src_entity_id"].str.startswith('A_')) & df_ref["src_entity_id"].notnull()].iloc[0]
        if len(ser_src_entity_id):
            src_entity_id = ser_src_entity_id.iloc[0]
            if src_entity_id:
                repo_id = Entity_search.get_first_match_or_none(r'(?<=_)\d+', str(src_entity_id))
        if repo_id is None:
            repo_id = get_repo_id_by_repo_key(repo_key, df_repo_i_pr_rec_cnt)

        # 存储结果
        issue_referencing_metrics_list.append({
            'repo_id': repo_id,
            'repo_key': repo_key,
            'issue_pr_active_id_count': issue_pr_active_id_count,
            'comment_related_id_count': comment_related_id_count,
            'comment_body_ref_count': comment_body_ref_count,
            'total_ref_count': total_ref_count,
            'active_issue_num': active_issue_num,
            'comment_per_issue': comment_per_issue,
            'referencing_per_comment': referencing_per_comment
        })

    if not issue_referencing_metrics_list:
        logger.warning("没有找到有效的项目数据来计算议题指标")
        return None

    df_issue_referencing_metrics = pd.DataFrame(issue_referencing_metrics_list)

    # 计算描述性统计
    metrics_to_describe = ['active_issue_num', 'comment_per_issue', 'referencing_per_comment']
    descriptive_stats = {}

    for metric in metrics_to_describe:
        if metric in df_issue_referencing_metrics.columns:
            stats = df_issue_referencing_metrics[metric].describe()
            descriptive_stats[metric] = {
                'mean': stats['mean'],
                'median': stats['50%'],
                'min': stats['min'],
                'max': stats['max'],
                'std': stats['std']
            }

    # 保存结果
    github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
    output_dir = os.path.join(github_osdb_data_dir, "analysis_results/issue_referencing_metrics")
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细数据
    detailed_path = os.path.join(output_dir, "df_issue_referencing_metrics.csv")
    df_issue_referencing_metrics.to_csv(detailed_path, index=False, encoding='utf-8')

    # 保存描述性统计
    stats_path = os.path.join(output_dir, "df_issue_referencing_metrics_stats.csv")
    df_stats = pd.DataFrame.from_dict(descriptive_stats, orient='index')
    df_stats.to_csv(stats_path, index=True, encoding='utf-8')

    # 打印统计信息
    logger.info("议题指标描述性统计:")
    for metric, stats in descriptive_stats.items():
        logger.info(f"  {metric}: 平均值={stats['mean']:.4f}, 中位数={stats['median']:.4f}, "
                    f"最小值={stats['min']:.4f}, 最大值={stats['max']:.4f}, 标准差={stats['std']:.4f}")

    return df_issue_referencing_metrics


def analyze_self_ref_time_evolution(df_self_ref_ratio, df_repo_i_pr_rec_cnt):
    """
    分析自引率时间演化特征
    外引比率随项目年龄的散点图
    """
    logger.info("c. analyze_self_ref_time_evolution...")

    # 确保数据类型一致
    df_repo_i_pr_rec_cnt["repo_id"] = df_repo_i_pr_rec_cnt["repo_id"].astype(str)
    df_self_ref_ratio["repo_id"] = df_self_ref_ratio["repo_id"].astype(str)

    # 过滤掉repo_name和repo_created_at同时为空的记录
    mask_not_null = df_repo_i_pr_rec_cnt["repo_name"].notna() & df_repo_i_pr_rec_cnt["repo_created_at"].notna()
    df_repo_info_clean = df_repo_i_pr_rec_cnt[mask_not_null].copy()

    # 检查去除空值后的唯一性
    if df_repo_info_clean["repo_name"].duplicated().any():
        logger.warning("去重后repo_name仍存在重复值，保留第一条记录")
        df_repo_info_clean = df_repo_info_clean.drop_duplicates(subset=['repo_name'], keep='first')

    if df_repo_info_clean["repo_id"].duplicated().any():
        logger.warning("去重后repo_id仍存在重复值，保留第一条记录")
        df_repo_info_clean = df_repo_info_clean.drop_duplicates(subset=['repo_id'], keep='first')

    logger.info(f"有效项目数量（有创建时间）: {len(df_repo_info_clean)}")

    # 合并自引率和项目信息
    df_merged = pd.merge(df_self_ref_ratio, df_repo_info_clean, on=['repo_id', 'repo_name'], how='inner')

    if len(df_merged) == 0:
        logger.warning("没有找到匹配的项目数据来分析自引率时间演化")
        return None

    logger.info(f"成功匹配 {len(df_merged)} 个项目的数据")

    # 计算项目年龄
    def calculate_project_age(created_at_str, reference_year=2023):
        """计算项目年龄（到参考年份年底的年数）"""
        try:
            # 转换时间戳，统一时区处理
            created_at = pd.to_datetime(created_at_str)

            # 如果时间戳带有时区，转换为UTC并移除时区信息
            if created_at.tz is not None:
                created_at = created_at.tz_convert('UTC').tz_localize(None)

            # 创建参考时间（2023年底），确保不带时区
            end_of_year = pd.Timestamp(f'{reference_year}-12-31')

            # 计算天数差
            age_days = (end_of_year - created_at).days
            age_years = age_days / 365.25  # 转换为年
            return age_years
        except Exception as e:
            logger.error(f"计算项目年龄时出错: {e}, 原始数据: {created_at_str}")
            return None

    df_merged['project_age_years'] = df_merged['repo_created_at'].apply(
        lambda x: calculate_project_age(x, reference_year=2023)
    )

    # 移除无法计算年龄的记录
    df_merged = df_merged.dropna(subset=['project_age_years'])

    if len(df_merged) == 0:
        logger.warning("没有有效的项目年龄数据")
        return None

    # 计算外引比率
    df_merged['external_ref_ratio'] = 1 - df_merged['self_ref_ratio']

    # 保存合并后的数据
    github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
    output_dir = os.path.join(github_osdb_data_dir, "analysis_results/self_ref_evolution")
    os.makedirs(output_dir, exist_ok=True)

    merged_path = os.path.join(output_dir, "df_self_ref_evolution_merged.csv")
    df_merged.to_csv(merged_path, index=False, encoding='utf-8')

    # 绘制外引比率随项目年龄的散点图
    try:

        plt.figure(figsize=(12, 8))

        # 创建散点图
        scatter = sns.scatterplot(
            data=df_merged,
            x='project_age_years',
            y='external_ref_ratio',
            size='i_pr_rec_cnt',  # 用活动记录数量作为点的大小
            sizes=(50, 500),  # 点的大小范围
            alpha=0.7,
            hue='external_ref_ratio',  # 根据外引比率着色
            palette='viridis'
        )

        # 添加回归线
        sns.regplot(
            data=df_merged,
            x='project_age_years',
            y='external_ref_ratio',
            scatter=False,
            color='red',
            line_kws={'linestyle': '--', 'alpha': 0.7, 'linewidth': 2}
        )

        # 使用英文标签
        plt.xlabel('Project Age (years)', fontsize=12)
        plt.ylabel('External Reference Ratio', fontsize=12)
        plt.title('External Reference Ratio vs Project Age for DBMS Projects (2023)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')

        # 添加图例
        plt.legend(title='Activity Records', bbox_to_anchor=(1.05, 1), loc='upper left')

        # 添加项目名称标签（仅标注前N个最大的点）
        n_labels = min(10, len(df_merged))
        largest_points = df_merged.nlargest(n_labels, 'i_pr_rec_cnt')

        for idx, row in largest_points.iterrows():
            plt.annotate(
                row['repo_name'].split('/')[-1],  # 只显示项目名，不显示组织名
                xy=(row['project_age_years'], row['external_ref_ratio']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

        plt.tight_layout()

        # 保存图表
        plot_path = os.path.join(output_dir, "external_ref_vs_project_age_scatter.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"散点图已保存至: {plot_path}")

        # 计算相关性
        correlation, p_value = stats.pearsonr(
            df_merged['project_age_years'],
            df_merged['external_ref_ratio']
        )

        # 计算回归线参数
        slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
            df_merged['project_age_years'],
            df_merged['external_ref_ratio']
        )

        logger.info(f"项目年龄与外引比率的相关系数: {correlation:.4f} (p={p_value:.4f})")
        logger.info(f"回归分析: 斜率={slope:.4f}, 截距={intercept:.4f}, R²={r_value ** 2:.4f}")

        # 保存相关性结果
        corr_data = {
            'correlation_coefficient': correlation,
            'p_value': p_value,
            'regression_slope': slope,
            'regression_intercept': intercept,
            'r_squared': r_value ** 2,
            'standard_error': std_err,
            'sample_size': len(df_merged)
        }
        corr_df = pd.DataFrame([corr_data])
        corr_path = os.path.join(output_dir, "correlation_analysis.csv")
        corr_df.to_csv(corr_path, index=False, encoding='utf-8')

        # 创建分组统计表（按项目年龄分组）
        # 使用等宽分箱或基于分位数的分箱
        if len(df_merged) >= 5:
            try:
                df_merged['age_group'] = pd.qcut(
                    df_merged['project_age_years'],
                    q=5,
                    duplicates='drop'  # 处理重复值
                )
            except:
                # 如果分位数分箱失败，使用等宽分箱
                df_merged['age_group'] = pd.cut(
                    df_merged['project_age_years'],
                    bins=5,
                    include_lowest=True
                )

            age_group_stats = df_merged.groupby('age_group').agg({
                'external_ref_ratio': ['mean', 'median', 'std', 'count'],
                'self_ref_ratio': ['mean', 'median', 'std'],
                'project_age_years': 'mean'
            }).round(4)

            # 创建自定义的标签显示格式（保留3位小数）
            def format_interval(interval):
                if isinstance(interval, pd.Interval):
                    return f'[{interval.left:.3f}, {interval.right:.3f})'
                return str(interval)

            # 创建格式化后的标签列
            df_merged['age_group_formatted'] = df_merged['age_group'].apply(format_interval)

            age_group_stats.columns = ['_'.join(col).strip() for col in age_group_stats.columns.values]
            age_group_stats = age_group_stats.rename(columns={
                'external_ref_ratio_mean': 'external_ref_ratio_mean',
                'external_ref_ratio_median': 'external_ref_ratio_median',
                'external_ref_ratio_std': 'external_ref_ratio_std',
                'external_ref_ratio_count': 'project_count',
                'ratio_mean': 'self_ref_ratio_mean',
                'ratio_median': 'self_ref_ratio_median',
                'ratio_std': 'self_ref_ratio_std',
                'project_age_years_mean': 'average_project_age'
            })

            age_group_path = os.path.join(output_dir, "age_group_statistics.csv")
            age_group_stats.to_csv(age_group_path, encoding='utf-8')

            logger.info("按年龄分组统计:")
            logger.info(f"\n{age_group_stats}")

            # 绘制年龄分组箱线图 - 使用格式化后的标签
            plt.figure(figsize=(10, 6))
            ax = sns.boxplot(data=df_merged, x='age_group_formatted', y='external_ref_ratio')
            plt.xlabel('Project Age Group (years)', fontsize=12)
            plt.ylabel('External Reference Ratio', fontsize=12)
            plt.title('External Reference Ratio Distribution by Project Age Group', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)

            # 在每个箱子上标注项目数量
            for i, group in enumerate(df_merged['age_group_formatted'].dropna().unique()):
                group_data = df_merged[df_merged['age_group_formatted'] == group]
                count = len(group_data)
                plt.text(i, group_data['external_ref_ratio'].max() + 0.02,
                         f'n={count}',
                         ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            boxplot_path = os.path.join(output_dir, "external_ref_by_age_group_boxplot.png")
            plt.savefig(boxplot_path, dpi=300)
            plt.close()

            logger.info(f"年龄分组箱线图已保存至: {boxplot_path}")

        # 分析高自引率和低自引率的项目特征
        if len(df_merged) >= 5:
            high_self_ref = df_merged.nlargest(min(5, len(df_merged)), 'self_ref_ratio')
            low_self_ref = df_merged.nsmallest(min(5, len(df_merged)), 'self_ref_ratio')

            high_self_ref_summary = high_self_ref[['repo_name', 'self_ref_ratio', 'project_age_years', 'i_pr_rec_cnt']]
            low_self_ref_summary = low_self_ref[['repo_name', 'self_ref_ratio', 'project_age_years', 'i_pr_rec_cnt']]

            logger.info("Top 5 projects with highest self-reference ratio:")
            for _, row in high_self_ref_summary.iterrows():
                logger.info(
                    f"  {row['repo_name']}: self_ref_ratio={row['self_ref_ratio']:.3f}, age={row['project_age_years']:.1f} years, activity_records={row['i_pr_rec_cnt']}")

            logger.info("Top 5 projects with lowest self-reference ratio:")
            for _, row in low_self_ref_summary.iterrows():
                logger.info(
                    f"  {row['repo_name']}: self_ref_ratio={row['self_ref_ratio']:.3f}, age={row['project_age_years']:.1f} years, activity_records={row['i_pr_rec_cnt']}")

            # 保存极端案例分析
            extremes_path = os.path.join(output_dir, "self_ref_extreme_cases.csv")
            extremes_df = pd.concat([
                high_self_ref_summary.assign(category='high_self_ref'),
                low_self_ref_summary.assign(category='low_self_ref')
            ])
            extremes_df.to_csv(extremes_path, index=False, encoding='utf-8')

    except Exception as e:
        logger.error(f"绘制图表或计算统计量时出错: {str(e)}")
        traceback.print_exc()

    # 分析自引率与其他指标的关系
    try:
        # 计算自引率与活动记录数的关系
        correlation_with_activity, p_activity = stats.pearsonr(
            df_merged['self_ref_ratio'],
            df_merged['i_pr_rec_cnt']
        )

        logger.info(
            f"Correlation between self-reference ratio and activity records: {correlation_with_activity:.4f} (p={p_activity:.4f})")

        # 创建多关系散点图矩阵
        plt.figure(figsize=(12, 10))
        scatter_vars = ['project_age_years', 'self_ref_ratio', 'i_pr_rec_cnt']
        scatter_df = df_merged[scatter_vars].copy()
        scatter_df.columns = ['Project Age (years)', 'Self-Reference Ratio', 'Activity Records']

        # 对活动记录数取对数以便更好地可视化
        scatter_df['log10(Activity Records)'] = np.log10(scatter_df['Activity Records'] + 1)

        scatter_matrix = pd.plotting.scatter_matrix(
            scatter_df[['Project Age (years)', 'Self-Reference Ratio', 'log10(Activity Records)']],
            figsize=(12, 10),
            diagonal='hist',
            alpha=0.5
        )

        plt.suptitle('DBMS Project Feature Relationship Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()

        scatter_matrix_path = os.path.join(output_dir, "project_features_scatter_matrix.png")
        plt.savefig(scatter_matrix_path, dpi=300)
        plt.close()

        logger.info(f"特征关系矩阵图已保存至: {scatter_matrix_path}")

    except Exception as e:
        logger.error(f"分析其他关系时出错: {str(e)}")

    logger.info("自引率时间演化分析完成。")
    return df_merged


def analyze_degree_distribution(G, output_dir=None, log_log_plot=True, fit_power_law=True, only_dbms_repo=False):
    """
    分析图的度分布与无标度特性

    参数:
    G: 网络图对象 (NetworkX Graph)
    output_dir: 输出目录路径
    log_log_plot: 是否绘制双对数坐标图
    fit_power_law: 是否拟合幂律分布

    返回:
    dict: 包含度分布统计信息的字典
    """

    logger.info("开始分析度分布与无标度特性...")

    if output_dir is None:
        github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
        output_dir = os.path.join(github_osdb_data_dir, f"analysis_results/degree_distribution{'_only_dbms_repo' if only_dbms_repo else ''}")

    os.makedirs(output_dir, exist_ok=True)

    # 辅助函数：将NumPy类型转换为Python原生类型
    def convert_to_serializable(obj):
        """将NumPy类型转换为Python原生类型以便JSON序列化"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # 1. 计算度分布
    degrees = [deg for _, deg in G.degree()]
    degree_counts = Counter(degrees)

    # 转换为DataFrame便于分析
    degree_df = pd.DataFrame({
        'degree': list(degree_counts.keys()),
        'frequency': list(degree_counts.values()),
        'probability': [count / len(degrees) for count in degree_counts.values()]
    })

    # 按度值排序
    degree_df = degree_df.sort_values('degree').reset_index(drop=True)

    # 2. 计算基本统计量
    degree_stats = {
        'min_degree': float(np.min(degrees)),
        'max_degree': float(np.max(degrees)),
        'mean_degree': float(np.mean(degrees)),
        'median_degree': float(np.median(degrees)),
        'std_degree': float(np.std(degrees)),
        'skewness': float(stats.skew(degrees)),
        'kurtosis': float(stats.kurtosis(degrees)),
        'num_nodes': int(len(degrees)),
        'num_edges': int(G.number_of_edges())
    }

    # 3. 检查是否符合幂律分布
    if fit_power_law and len(degree_df) >= 10:
        try:
            # 选择度值大于等于x_min的数据点
            degree_values = np.array(degree_df['degree'])
            frequency_values = np.array(degree_df['frequency'])

            # 排除度为0的点
            mask = degree_values > 0
            degree_values = degree_values[mask]
            frequency_values = frequency_values[mask]

            if len(degree_values) >= 5:
                # 拟合幂律分布: P(k) ~ k^(-γ)
                def power_law(x, gamma, C):
                    return C * np.power(x, -gamma)

                # 使用非线性最小二乘法拟合
                try:
                    popt, pcov = optimize.curve_fit(
                        power_law,
                        degree_values,
                        frequency_values / np.sum(frequency_values),
                        p0=[2.0, 1.0],  # 初始猜测值
                        bounds=([1.0, 0.0], [5.0, 10.0])  # 参数范围
                    )

                    gamma_fit = float(popt[0])
                    C_fit = float(popt[1])

                    # 计算R²
                    y_pred = power_law(degree_values, gamma_fit, C_fit)
                    y_true = frequency_values / np.sum(frequency_values)
                    ss_res = np.sum((y_true - y_pred) ** 2)  # 残差平方和
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总离差平方和
                    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0  # 回归平方和/总离差平方和

                    degree_stats.update({
                        'power_law_gamma': gamma_fit,
                        'power_law_C': C_fit,
                        'power_law_r_squared': r_squared,
                        'is_scale_free': r_squared > 0.8  # R²阈值判断
                    })

                    logger.info(f"幂律拟合结果: γ = {gamma_fit:.4f}, C = {C_fit:.4f}, R² = {r_squared:.4f}")

                except Exception as e:
                    logger.warning(f"幂律分布拟合失败: {str(e)}")
                    degree_stats.update({
                        'power_law_gamma': None,
                        'power_law_C': None,
                        'power_law_r_squared': None,
                        'is_scale_free': False
                    })
            else:
                logger.warning("有效数据点不足，无法进行幂律分布拟合")
                degree_stats.update({
                    'power_law_gamma': None,
                    'power_law_C': None,
                    'power_law_r_squared': None,
                    'is_scale_free': False
                })

        except Exception as e:
            logger.error(f"检查幂律分布时出错: {str(e)}")
            traceback.print_exc()

    # 4. 计算累积度分布
    degree_df['cumulative_probability'] = degree_df['probability'].cumsum()
    degree_df['complementary_cumulative'] = 1 - degree_df['cumulative_probability'] + degree_df['probability']

    # 5. 绘制度分布图
    plt.figure(figsize=(15, 12))

    # 子图1: 原始度分布直方图
    plt.subplot(2, 2, 1)
    plt.hist(degrees, bins=min(50, len(set(degrees))), alpha=0.7, edgecolor='black')
    plt.xlabel('Degree (k)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Degree Distribution Histogram', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f"Nodes: {int(degree_stats['num_nodes'])}\n"
    stats_text += f"Mean: {degree_stats['mean_degree']:.2f}\n"
    stats_text += f"Median: {degree_stats['median_degree']:.2f}\n"
    stats_text += f"Max: {int(degree_stats['max_degree'])}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)

    # 子图2: 对数-对数坐标下的度分布
    plt.subplot(2, 2, 2)
    if log_log_plot:
        # 筛选掉频率为0的点
        plot_df = degree_df[degree_df['frequency'] > 0].copy()

        if len(plot_df) > 0:
            plt.loglog(plot_df['degree'], plot_df['frequency'], 'bo', alpha=0.7, markersize=6)
            plt.xlabel('Degree (k) - log scale', fontsize=12)
            plt.ylabel('Frequency - log scale', fontsize=12)
            plt.title('Degree Distribution (Log-Log Plot)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, which='both')

            # 如果拟合了幂律分布，添加拟合线
            if 'power_law_gamma' in degree_stats and degree_stats['power_law_gamma'] is not None:
                k_min = np.min(plot_df['degree'])
                k_max = np.max(plot_df['degree'])
                k_range = np.logspace(np.log10(k_min), np.log10(k_max), 100)
                y_fit = degree_stats['power_law_C'] * np.power(k_range, -degree_stats['power_law_gamma'])
                plt.loglog(k_range, y_fit * len(degrees), 'r-', linewidth=2,
                           label=f'Power Law Fit (γ={degree_stats["power_law_gamma"]:.2f})')
                plt.legend()

    # 子图3: 累积度分布
    plt.subplot(2, 2, 3)
    plt.plot(degree_df['degree'], degree_df['cumulative_probability'], 'g-', linewidth=2)
    plt.xlabel('Degree (k)', fontsize=12)
    plt.ylabel('Cumulative Probability P(K ≤ k)', fontsize=12)
    plt.title('Cumulative Degree Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 子图4: 补充累积度分布（对数坐标）
    plt.subplot(2, 2, 4)
    ccdf_df = degree_df[degree_df['complementary_cumulative'] > 0].copy()
    if len(ccdf_df) > 0:
        plt.loglog(ccdf_df['degree'], ccdf_df['complementary_cumulative'], 'mo-', linewidth=2, markersize=6)
        plt.xlabel('Degree (k) - log scale', fontsize=12)
        plt.ylabel('P(K > k) - log scale', fontsize=12)
        plt.title('Complementary Cumulative Distribution (CCDF)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # 保存图形
    plot_path = os.path.join(output_dir, "degree_distribution_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"度分布图已保存至: {plot_path}")

    # 6. 保存数据
    # 度分布数据
    degree_dist_path = os.path.join(output_dir, "degree_distribution_data.csv")
    degree_df.to_csv(degree_dist_path, index=False, encoding='utf-8')

    # 度统计信息
    stats_path = os.path.join(output_dir, "degree_statistics.csv")
    stats_df = pd.DataFrame([degree_stats])
    stats_df.to_csv(stats_path, index=False, encoding='utf-8')

    # 7. 生成详细分析报告
    analysis_report = {
        'graph_basic_info': {
            'number_of_nodes': int(G.number_of_nodes()),
            'number_of_edges': int(G.number_of_edges()),
            'density': float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
            'is_connected': bool(nx.is_connected(G))
        },
        'degree_distribution': {
            'min': float(degree_stats['min_degree']),
            'max': float(degree_stats['max_degree']),
            'mean': float(degree_stats['mean_degree']),
            'median': float(degree_stats['median_degree']),
            'std': float(degree_stats['std_degree']),
            'skewness': float(degree_stats['skewness']),
            'kurtosis': float(degree_stats['kurtosis'])
        },
        'scale_free_properties': {
            'gamma_estimate': float(degree_stats.get('power_law_gamma')) if degree_stats.get(
                'power_law_gamma') is not None else None,
            'r_squared': float(degree_stats.get('power_law_r_squared')) if degree_stats.get(
                'power_law_r_squared') is not None else None,
            'is_scale_free': bool(degree_stats.get('is_scale_free', False))
        }
    }

    # 计算度同配性
    try:
        if G.number_of_edges() > 0:
            assortativity = float(nx.degree_assortativity_coefficient(G))
            analysis_report['degree_correlation'] = {
                'assortativity': assortativity
            }
        else:
            analysis_report['degree_correlation'] = {
                'assortativity': None
            }
    except Exception as e:
        logger.warning(f"计算度同配性时出错: {str(e)}")
        analysis_report['degree_correlation'] = {
            'assortativity': None
        }

    # 确保所有值为可序列化的Python原生类型
    analysis_report = convert_to_serializable(analysis_report)

    # 保存报告
    report_path = os.path.join(output_dir, "degree_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=4, ensure_ascii=False)

    # 8. 打印关键发现
    logger.info("=" * 60)
    logger.info("度分布分析关键发现:")
    logger.info(f"1. 网络基本信息:")
    logger.info(f"   - 节点数: {int(G.number_of_nodes())}")
    logger.info(f"   - 边数: {int(G.number_of_edges())}")
    logger.info(f"   - 平均度: {degree_stats['mean_degree']:.2f}")

    logger.info(f"2. 度分布统计:")
    logger.info(f"   - 最小度: {int(degree_stats['min_degree'])}")
    logger.info(f"   - 最大度: {int(degree_stats['max_degree'])}")
    logger.info(f"   - 中位数: {degree_stats['median_degree']:.2f}")
    logger.info(f"   - 偏度: {degree_stats['skewness']:.4f} (正值表示右偏)")

    if 'power_law_gamma' in degree_stats and degree_stats['power_law_gamma'] is not None:
        logger.info(f"3. 无标度特性:")
        logger.info(f"   - 幂律指数 γ = {degree_stats['power_law_gamma']:.4f}")
        logger.info(f"   - 拟合优度 R² = {degree_stats['power_law_r_squared']:.4f}")
        if degree_stats.get('is_scale_free'):
            logger.info(f"   - 判断: 具有无标度特性")
        else:
            logger.info(f"   - 判断: 不具有显著无标度特性")

    # 计算度同配性
    try:
        if G.number_of_edges() > 0:
            assortativity = nx.degree_assortativity_coefficient(G)
            logger.info(f"4. 度相关性:")
            logger.info(f"   - 同配系数: {float(assortativity):.4f}")
            if assortativity > 0:
                logger.info(f"   - 同配网络: 高度节点倾向于连接其他高度节点")
            elif assortativity < 0:
                logger.info(f"   - 异配网络: 高度节点倾向于连接低度节点")
            else:
                logger.info(f"   - 随机网络: 无明显的度相关性")
    except Exception as e:
        logger.warning(f"计算度同配性时出错: {str(e)}")

    logger.info("=" * 60)
    logger.info("度分布与无标度特性分析完成。")

    return degree_stats


def calculate_centrality_measures(G, output_dir=None, top_k=20, include_weighted=True, use_largest_component=True, only_dbms_repo=False):
    """
    计算图的中心性指标并识别关键节点

    参数:
    G: 网络图对象 (NetworkX Graph)
    output_dir: 输出目录路径
    top_k: 输出前k个关键节点
    include_weighted: 是否计算加权中心性
    use_largest_component: 是否在最大连通子图上计算接近中心性等需要连通图的指标

    返回:
    dict: 包含中心性分析结果的字典
    """
    import numpy as np

    logger.info("开始计算中心性指标并识别关键节点...")

    if output_dir is None:
        github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
        output_dir = os.path.join(github_osdb_data_dir, f"analysis_results/centrality_analysis{'_only_dbms_repo' if only_dbms_repo else ''}")

    os.makedirs(output_dir, exist_ok=True)

    # 确保图有节点
    if G.number_of_nodes() == 0:
        logger.warning("图为空，无法计算中心性指标")
        return {}

    # 获取最大连通子图（仅用于需要连通图的中心性计算）
    if use_largest_component and not nx.is_connected(G):
        # 获取所有连通分量
        connected_components = list(nx.connected_components(G))
        if connected_components:
            # 找到最大的连通分量
            largest_component = max(connected_components, key=len)
            G_lcc = G.subgraph(largest_component).copy()

            logger.info(f"识别最大连通子图(LCC)用于需要连通图的中心性计算")
            logger.info(f"   - 原始图节点数: {G.number_of_nodes()}")
            logger.info(
                f"   - LCC节点数: {G_lcc.number_of_nodes()} (覆盖率: {G_lcc.number_of_nodes() / G.number_of_nodes():.2%})")
            logger.info(f"   - LCC边数: {G_lcc.number_of_edges()}")
        else:
            G_lcc = G
            logger.warning("图没有连通分量，使用原图计算中心性")
    else:
        G_lcc = G
        logger.info(f"使用完整图计算中心性指标 (节点数: {G.number_of_nodes()})")

    # 准备存储结果的字典
    centrality_results = {}

    # 1. 计算各种中心性指标
    try:
        # 度中心性 - 在原始图G上计算（包含所有节点）
        degree_centrality = nx.degree_centrality(G)
        centrality_results['degree_centrality'] = degree_centrality
        logger.info("计算度中心性（基于完整图）")

        # 接近中心性 - 需要在连通图上计算，使用LCC
        if use_largest_component:
            if nx.is_connected(G_lcc):
                closeness_centrality_lcc = nx.closeness_centrality(G_lcc)
                # 将LCC的结果扩展到整个图
                closeness_centrality = {}
                for node in G.nodes():
                    if node in closeness_centrality_lcc:
                        closeness_centrality[node] = closeness_centrality_lcc[node]
                    else:
                        closeness_centrality[node] = 0.0  # 非LCC节点的接近中心性设为0
                centrality_results['closeness_centrality'] = closeness_centrality
                logger.info("计算接近中心性（基于最大连通子图，非LCC节点设为0）")
            else:
                logger.warning("LCC不连通，无法计算接近中心性")
                closeness_centrality = {}
        elif nx.is_connected(G):
            closeness_centrality = nx.closeness_centrality(G)
            centrality_results['closeness_centrality'] = closeness_centrality
            logger.info("计算接近中心性（基于完整图）")
        else:
            logger.warning("图不连通，无法计算接近中心性")
            closeness_centrality = {}

        # 介数中心性 - 在原始图G上计算（包含所有节点）
        if G.number_of_nodes() < 1000:  # 对于大型网络，使用近似算法
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
            logger.info("使用精确算法计算介数中心性（基于完整图）")
        else:
            logger.info("节点数过多，使用近似算法计算介数中心性（基于完整图）")
            betweenness_centrality = nx.betweenness_centrality(G, k=100)
        centrality_results['betweenness_centrality'] = betweenness_centrality

        # 特征向量中心性 - 在原始图G上计算（包含所有节点）
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=500, tol=1e-06)
            centrality_results['eigenvector_centrality'] = eigenvector_centrality
            logger.info("计算特征向量中心性（基于完整图）")
        except Exception as e:
            logger.warning(f"计算特征向量中心性失败: {str(e)}")
            eigenvector_centrality = {}

        # PageRank中心性 - 在原始图G上计算（包含所有节点）
        pagerank = nx.pagerank(G, alpha=0.85)
        centrality_results['pagerank'] = pagerank
        logger.info("计算PageRank中心性（基于完整图）")

        # 加权中心性（如果图有权重）
        if include_weighted:
            # 检查图中是否有边权重
            has_weight = False
            for _, _, edge_data in G.edges(data=True):
                if 'weight' in edge_data:
                    has_weight = True
                    break

            if has_weight:
                try:
                    # 加权度中心性 - 在原始图G上计算
                    weighted_degree_centrality = {}
                    for node in G.nodes():
                        weighted_degree = sum([d.get('weight', 1) for _, _, d in G.edges(node, data=True)])
                        weighted_degree_centrality[node] = weighted_degree / (G.number_of_nodes() - 1)
                    centrality_results['weighted_degree_centrality'] = weighted_degree_centrality

                    # 加权PageRank - 在原始图G上计算
                    weighted_pagerank = nx.pagerank(G, alpha=0.85, weight='weight')
                    centrality_results['weighted_pagerank'] = weighted_pagerank

                    logger.info("计算加权中心性指标（基于完整图）")
                except Exception as e:
                    logger.warning(f"计算加权中心性失败: {str(e)}")
            else:
                logger.info("图中没有边权重，跳过加权中心性计算")

    except Exception as e:
        logger.error(f"计算中心性指标时出错: {str(e)}")
        traceback.print_exc()
        return {}

    # 2. 创建中心性DataFrame（基于原图所有节点）
    centrality_df = pd.DataFrame(index=list(G.nodes()))

    # 对于原图中的每个节点，使用计算的中心性值
    for centrality_name, centrality_dict in centrality_results.items():
        centrality_series = pd.Series(index=G.nodes(), dtype=float)
        for node in G.nodes():
            if node in centrality_dict:
                centrality_series[node] = centrality_dict[node]
            else:
                centrality_series[node] = 0.0  # 对于未计算的节点（如非LCC节点的接近中心性），设为0
        centrality_df[centrality_name] = centrality_series
    # 3. 计算综合中心性得分
    try:
        # 标准化各中心性指标（只对有效值进行标准化）
        centrality_normalized = centrality_df.copy()
        for col in centrality_df.columns:
            valid_values = centrality_df[col].dropna()
            if len(valid_values) > 1:
                # 最小-最大归一化
                min_val = valid_values.min()
                max_val = valid_values.max()
                if max_val > min_val:
                    centrality_normalized[col] = (centrality_df[col] - min_val) / (max_val - min_val)
                else:
                    centrality_normalized[col] = 0.5  # 所有值相等
            elif len(valid_values) == 1:
                centrality_normalized[col] = 1.0  # 只有一个有效值

        # 计算综合中心性得分（各中心性的平均值，跳过NaN）
        centrality_normalized['composite_centrality'] = centrality_normalized.mean(axis=1, skipna=True)

        # 添加到原DataFrame
        centrality_df['composite_centrality'] = centrality_normalized['composite_centrality']

        logger.info("计算综合中心性得分（基于完整图）")
    except Exception as e:
        logger.warning(f"计算综合中心性得分失败: {str(e)}")
        centrality_df['composite_centrality'] = np.nan

    # 4. 添加节点属性信息
    node_attrs = {}
    for node in G.nodes():
        attrs = G.nodes[node]
        for attr_name, attr_value in attrs.items():
            if attr_name not in node_attrs:
                node_attrs[attr_name] = {}
            node_attrs[attr_name][node] = attr_value

    # 将节点属性添加到DataFrame
    for attr_name, attr_dict in node_attrs.items():
        centrality_df[attr_name] = centrality_df.index.map(attr_dict)

    # 5. 标记节点是否在最大连通子图中（用于可视化区分）
    if use_largest_component:
        lcc_nodes = set(G_lcc.nodes())
        centrality_df['in_lcc'] = centrality_df.index.isin(lcc_nodes)

        # 计算LCC覆盖率统计
        lcc_coverage = {
            'total_nodes': len(centrality_df),
            'lcc_nodes': len(lcc_nodes),
            'coverage_ratio': len(lcc_nodes) / len(centrality_df) if len(centrality_df) > 0 else 0,
            'non_lcc_nodes': len(centrality_df) - len(lcc_nodes)
        }
        logger.info(f"LCC覆盖率: {lcc_coverage['lcc_nodes']}/{lcc_coverage['total_nodes']} "
                    f"({lcc_coverage['coverage_ratio']:.2%})")
    else:
        # 如果不使用LCC，则所有节点都视为"在LCC中"
        centrality_df['in_lcc'] = True

    # 6. 识别关键节点（基于所有节点）
    key_nodes_analysis = {}

    # 使用所有节点的数据
    centrality_df_all = centrality_df

    for centrality_name in centrality_df_all.columns:
        if centrality_name not in node_attrs.keys() and centrality_name not in ['in_lcc',
                                                                                'composite_centrality']:  # 跳过节点属性列
            # 获取前top_k个节点
            sorted_nodes = centrality_df_all[centrality_name].sort_values(ascending=False)
            top_nodes = sorted_nodes.head(top_k)

            # 获取节点详细信息
            key_nodes_info = []
            for node_id, centrality_value in top_nodes.items():
                node_info = {
                    'node_id': node_id,
                    'centrality_value': float(centrality_value) if pd.notna(centrality_value) else None,
                    'in_lcc': centrality_df.loc[node_id, 'in_lcc'] if 'in_lcc' in centrality_df.columns else True
                }

                # 添加节点属性
                for attr_name in ['repo_name', 'repo_id', 'degree', 'node_type']:
                    if attr_name in centrality_df.columns and node_id in centrality_df.index:
                        attr_value = centrality_df.loc[node_id, attr_name]
                        node_info[attr_name] = attr_value if pd.notna(attr_value) else None

                key_nodes_info.append(node_info)

            # 计算统计信息（基于所有节点）
            if centrality_name in centrality_df.columns:
                valid_values = centrality_df[centrality_name].dropna()
                if len(valid_values) > 0:
                    stats = {
                        'mean': float(valid_values.mean()),
                        'median': float(valid_values.median()),
                        'std': float(valid_values.std()),
                        'max': float(valid_values.max()),
                        'min': float(valid_values.min())
                    }
                else:
                    stats = {
                        'mean': None, 'median': None, 'std': None,
                        'max': None, 'min': None
                    }

                key_nodes_analysis[centrality_name] = {
                    'top_nodes': key_nodes_info,
                    'statistics': stats,
                    'based_on_all_nodes': True  # 标记基于所有节点
                }

    # 7. 计算中心性指标间的相关性（基于所有节点）
    try:
        centrality_cols = [col for col in centrality_df.columns
                           if col not in node_attrs.keys() and col not in ['in_lcc', 'composite_centrality']]
        if len(centrality_cols) > 1:
            centrality_correlation = centrality_df[centrality_cols].corr()
            logger.info(f"计算{len(centrality_cols)}个中心性指标的相关性（基于完整图）")
        else:
            centrality_correlation = pd.DataFrame()
            logger.warning("中心性指标不足，无法计算相关性")
    except Exception as e:
        logger.warning(f"计算中心性相关性失败: {str(e)}")
        centrality_correlation = pd.DataFrame()

    # 8. 保存结果
    # 保存完整的中心性数据
    centrality_csv_path = os.path.join(output_dir, "centrality_data.csv")
    centrality_df.to_csv(centrality_csv_path, encoding='utf-8')
    logger.info(f"中心性数据已保存至: {centrality_csv_path}")

    # 保存关键节点分析结果
    key_nodes_json_path = os.path.join(output_dir, "key_nodes_analysis.json")
    with open(key_nodes_json_path, 'w', encoding='utf-8') as f:
        # 确保所有值都可序列化
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        json.dump(convert_for_json(key_nodes_analysis), f, indent=4, ensure_ascii=False)

    logger.info(f"关键节点分析已保存至: {key_nodes_json_path}")

    # 保存相关性矩阵
    if not centrality_correlation.empty:
        correlation_csv_path = os.path.join(output_dir, "centrality_correlation.csv")
        centrality_correlation.to_csv(correlation_csv_path, encoding='utf-8')
        logger.info(f"中心性相关性矩阵已保存至: {correlation_csv_path}")

    # 9. 可视化分析（保持不变，但使用所有节点的数据）
    plt.figure(figsize=(18, 14))

    # 子图1: 网络连通性可视化（保持不变）
    plt.subplot(3, 4, 1)

    if use_largest_component:
        try:
            # 使用弹簧布局算法可视化网络结构
            if G_lcc.number_of_nodes() <= 200:  # 对于较小的网络，直接布局
                pos = nx.spring_layout(G_lcc, seed=42)

                # 绘制边
                nx.draw_networkx_edges(G_lcc, pos, alpha=0.1, width=0.5, edge_color='gray')

                # 绘制节点（区分LCC和非LCC）
                lcc_nodes = list(G_lcc.nodes())

                # 绘制LCC节点
                if lcc_nodes:
                    nx.draw_networkx_nodes(G_lcc, pos, nodelist=lcc_nodes,
                                           node_color='red', node_size=50,
                                           alpha=0.8, edgecolors='black', linewidths=0.5)

                plt.title(f'Network Structure\n(Red: LCC Nodes, n={len(lcc_nodes)})',
                          fontsize=11)

                # 添加统计信息
                stats_text = f"Total Nodes: {len(G.nodes())}\n"
                stats_text += f"LCC Nodes: {len(lcc_nodes)}\n"
                stats_text += f"LCC Coverage: {len(lcc_nodes) / len(G.nodes()):.1%}"

                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            else:
                # 对于大型网络，使用简化的可视化
                # 创建连通分量大小的分布图
                connected_components = list(nx.connected_components(G))
                component_sizes = [len(comp) for comp in connected_components]
                component_sizes.sort(reverse=True)

                # 只显示前10个最大的连通分量
                display_sizes = component_sizes[:10]
                labels = [f'CC{i + 1}' for i in range(len(display_sizes))]

                # 使用不同颜色标记LCC
                colors = ['red'] + ['lightblue'] * (len(display_sizes) - 1)

                plt.bar(range(len(display_sizes)), display_sizes, color=colors, alpha=0.8)
                plt.xticks(range(len(display_sizes)), labels, rotation=45)
                plt.xlabel('Connected Components', fontsize=9)
                plt.ylabel('Number of Nodes', fontsize=9)
                plt.title('Connected Components Size Distribution\n(Red: Largest CC)',
                          fontsize=11)
                plt.grid(True, alpha=0.3, axis='y')

                # 添加尺寸标签
                for i, size in enumerate(display_sizes):
                    plt.text(i, size + (max(display_sizes) * 0.02), str(size),
                             ha='center', va='bottom', fontsize=8)

                # 添加统计信息
                stats_text = f"Total CCs: {len(component_sizes)}\n"
                stats_text += f"LCC Size: {component_sizes[0] if component_sizes else 0}\n"
                stats_text += f"Avg CC Size: {np.mean(component_sizes):.1f}"

                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.axis('on')

        except Exception as e:
            logger.warning(f"网络可视化失败: {str(e)}")
            # 回退到简单的条形图
            lcc_nodes = set(G_lcc.nodes())
            non_lcc_nodes = set(G.nodes()) - lcc_nodes

            categories = ['LCC Nodes', 'Non-LCC Nodes']
            counts = [len(lcc_nodes), len(non_lcc_nodes)]
            colors = ['red', 'lightgray']

            plt.bar(categories, counts, color=colors, alpha=0.8)
            plt.ylabel('Number of Nodes', fontsize=9)
            plt.title('Node Distribution\n(Largest Connected Component)', fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for i, count in enumerate(counts):
                plt.text(i, count + (max(counts) * 0.02), str(count),
                         ha='center', va='bottom', fontsize=9)

    else:
        # 不使用LCC的情况
        if G.number_of_nodes() <= 200:
            try:
                pos = nx.spring_layout(G, seed=42)
                nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, edge_color='gray')
                nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=30,
                                       alpha=0.6, edgecolors='black', linewidths=0.5)
                plt.title(f'Network Structure\n(Complete Graph, n={G.number_of_nodes()})',
                          fontsize=11)
                plt.axis('on')

                # 添加统计信息
                stats_text = f"Total Nodes: {G.number_of_nodes()}\n"
                stats_text += f"Total Edges: {G.number_of_edges()}\n"
                stats_text += f"Density: {nx.density(G):.4f}"

                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            except Exception as e:
                logger.warning(f"网络可视化失败: {str(e)}")
                # 简单的节点统计
                plt.text(0.5, 0.5, f"Nodes: {G.number_of_nodes()}\nEdges: {G.number_of_edges()}",
                         ha='center', va='center', transform=plt.gca().transAxes,
                         fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                plt.title('Network Information', fontsize=11)
                plt.axis('off')
        else:
            # 对于大型网络，显示基本统计
            plt.text(0.5, 0.5, f"Large Network:\n{G.number_of_nodes()} Nodes\n{G.number_of_edges()} Edges",
                     ha='center', va='center', transform=plt.gca().transAxes,
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.title('Network Information', fontsize=11)
            plt.axis('off')

    # 子图2-7: 各中心性指标分布（基于所有节点）
    centrality_cols = [col for col in centrality_df.columns
                       if col not in node_attrs.keys() and col not in ['in_lcc', 'composite_centrality'] and
                       centrality_df[col].notna().any()]

    if centrality_cols:
        # 选择前6个中心性指标进行可视化
        plot_cols = centrality_cols[:6]

        for i, col in enumerate(plot_cols, 2):
            plt.subplot(3, 4, i)

            # 绘制LCC节点和非LCC节点的分布
            if use_largest_component and 'in_lcc' in centrality_df.columns:
                lcc_values = centrality_df[centrality_df['in_lcc']][col].dropna()
                non_lcc_values = centrality_df[~centrality_df['in_lcc']][col].dropna()

                if len(lcc_values) > 0:
                    plt.hist(lcc_values, bins=20, alpha=0.7, color='red',
                             edgecolor='black', label=f'LCC Nodes (n={len(lcc_values)})')
                if len(non_lcc_values) > 0:
                    plt.hist(non_lcc_values, bins=20, alpha=0.3, color='gray',
                             edgecolor='black', label=f'Non-LCC Nodes (n={len(non_lcc_values)})')

                if len(lcc_values) > 0 or len(non_lcc_values) > 0:
                    plt.legend(fontsize=8)
            else:
                plt.hist(centrality_df[col].dropna(), bins=30, alpha=0.7,
                         edgecolor='black', color='blue', label=f'All Nodes (n={len(centrality_df[col].dropna())})')
                plt.legend(fontsize=8)

            plt.xlabel(col.replace('_', ' ').title(), fontsize=9)
            plt.ylabel('Frequency', fontsize=9)
            plt.title(f'{col.replace("_", " ").title()}\nDistribution', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

    # 子图8: 中心性相关性热图（基于所有节点）
    if not centrality_correlation.empty and len(centrality_correlation) > 1:
        plt.subplot(3, 4, 8)
        sns.heatmap(centrality_correlation, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, cbar_kws={'shrink': 0.8})
        correlation_title = 'Centrality Measures Correlation\n(All Nodes)'
        plt.title(correlation_title, fontsize=11, fontweight='bold')

    # 子图9: 综合中心性前10名节点（基于所有节点）
    if 'composite_centrality' in centrality_df.columns:
        plt.subplot(3, 4, 9)
        # 选择所有节点中综合中心性最高的前10个
        top_composite = centrality_df.nlargest(min(10, len(centrality_df)), 'composite_centrality')

        if 'repo_name' in top_composite.columns:
            labels = []
            for idx, row in top_composite.iterrows():
                repo_name = row['repo_name'] if pd.notna(row['repo_name']) else str(idx)
                if use_largest_component and 'in_lcc' in row:
                    repo_name = f"{repo_name}{'*' if row['in_lcc'] else ''}"
                labels.append(repo_name[:20] + '...' if len(repo_name) > 20 else repo_name)
        else:
            labels = [str(idx)[:15] + '...' for idx in top_composite.index]

        plt.barh(range(len(top_composite)), top_composite['composite_centrality'])
        plt.yticks(range(len(top_composite)), labels, fontsize=8)
        plt.xlabel('Composite Centrality Score', fontsize=9)
        plt.title(f'Top {len(top_composite)} Key Nodes\n(* indicates LCC)', fontsize=10, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')

    # 子图10: 节点度与综合中心性的散点图（基于所有节点）
    if 'degree' in centrality_df.columns and 'composite_centrality' in centrality_df.columns:
        plt.subplot(3, 4, 10)

        if use_largest_component and 'in_lcc' in centrality_df.columns:
            # 区分LCC和非LCC节点
            lcc_mask = centrality_df['in_lcc']

            # 获取LCC和非LCC数据
            lcc_data = centrality_df[lcc_mask]
            non_lcc_data = centrality_df[~lcc_mask]

            # 绘制LCC节点
            if len(lcc_data) > 0:
                valid_lcc = lcc_data.dropna(subset=['degree', 'composite_centrality'])
                if len(valid_lcc) > 0:
                    plt.scatter(valid_lcc['degree'], valid_lcc['composite_centrality'],
                                alpha=0.7, s=30, c='red', edgecolors='black', linewidth=0.5,
                                label=f'LCC Nodes (n={len(valid_lcc)})')

            # 绘制非LCC节点
            if len(non_lcc_data) > 0:
                valid_non_lcc = non_lcc_data.dropna(subset=['degree', 'composite_centrality'])
                if len(valid_non_lcc) > 0:
                    plt.scatter(valid_non_lcc['degree'], valid_non_lcc['composite_centrality'],
                                alpha=0.3, s=20, c='gray', edgecolors='black', linewidth=0.5,
                                label=f'Non-LCC Nodes (n={len(valid_non_lcc)})')

            # 显示图例
            if len(lcc_data) > 0 or len(non_lcc_data) > 0:
                plt.legend(fontsize=8)
        else:
            # 不使用LCC区分的情况
            valid_data = centrality_df.dropna(subset=['degree', 'composite_centrality'])
            if len(valid_data) > 0:
                plt.scatter(valid_data['degree'], valid_data['composite_centrality'],
                            alpha=0.6, s=20, edgecolors='black', linewidth=0.5,
                            label=f'All Nodes (n={len(valid_data)})')
                plt.legend(fontsize=8)

        plt.xlabel('Degree', fontsize=9)
        plt.ylabel('Composite Centrality', fontsize=9)
        plt.title('Degree vs Composite Centrality', fontsize=10)
        plt.grid(True, alpha=0.3)

        # 添加回归线（基于所有有效节点）
        valid_data = centrality_df.dropna(subset=['degree', 'composite_centrality'])
        if len(valid_data) > 1:
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_data['degree'], valid_data['composite_centrality']
                )
                x_range = np.linspace(valid_data['degree'].min(), valid_data['degree'].max(), 100)
                y_pred = slope * x_range + intercept
                plt.plot(x_range, y_pred, 'b-', linewidth=2,
                         label=f'Fit: r={r_value:.3f}')
                plt.legend(fontsize=8)
            except Exception as e:
                logger.debug(f"添加回归线失败: {str(e)}")

        # 子图11: LCC节点中心性对比（基于所有节点）
        if use_largest_component and 'in_lcc' in centrality_df.columns:
            plt.subplot(3, 4, 11)

            # 比较LCC节点和非LCC节点的综合中心性
            lcc_centralities = centrality_df[centrality_df['in_lcc']]['composite_centrality'].dropna()
            non_lcc_centralities = centrality_df[~centrality_df['in_lcc']]['composite_centrality'].dropna()

            # 检查是否有数据
            has_lcc_data = len(lcc_centralities) > 0
            has_non_lcc_data = len(non_lcc_centralities) > 0

            if has_lcc_data or has_non_lcc_data:
                data_to_plot = []
                labels = []
                colors = []

                if has_lcc_data:
                    data_to_plot.append(lcc_centralities)
                    labels.append(f'LCC\n(n={len(lcc_centralities)})')
                    colors.append('red')

                if has_non_lcc_data:
                    data_to_plot.append(non_lcc_centralities)
                    labels.append(f'Non-LCC\n(n={len(non_lcc_centralities)})')
                    colors.append('lightgray')

                # 绘制箱线图
                bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)

                # 设置箱体颜色
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # 设置须线颜色
                for element in ['whiskers', 'caps', 'medians']:
                    for line in bp[element]:
                        line.set_color('black')
                        line.set_linewidth(1)

                # 设置离群点
                for flier in bp['fliers']:
                    flier.set(marker='o', color='black', alpha=0.5, markersize=3)

                plt.ylabel('Composite Centrality', fontsize=9)

                # 添加标题（不带自动换行）
                plt.title('Centrality Comparison: LCC vs Non-LCC',
                          fontsize=10,
                          fontweight='bold',
                          pad=10)  # 稍微增加标题与图的间距

                plt.grid(True, alpha=0.3, axis='y')

                # 创建图例元素
                from matplotlib.patches import Patch
                legend_elements = []
                if has_lcc_data:
                    legend_elements.append(Patch(facecolor='red', alpha=0.7,
                                                 label=f'LCC (n={len(lcc_centralities)})'))
                if has_non_lcc_data:
                    legend_elements.append(Patch(facecolor='lightgray', alpha=0.7,
                                                 label=f'Non-LCC (n={len(non_lcc_centralities)})'))

                # 1. 先添加图例在图形内顶部
                if legend_elements:
                    # 将图例放在图形内顶部中央
                    legend = plt.legend(handles=legend_elements,
                                        loc='upper center',
                                        bbox_to_anchor=(0.5, 0.97),  # 在图形内顶部
                                        fontsize=8,
                                        framealpha=0.9,
                                        ncol=len(legend_elements),  # 根据元素数量决定列数
                                        borderaxespad=0.5,
                                        fancybox=True,
                                        shadow=False,
                                        handlelength=1.5,
                                        handletextpad=0.5,
                                        columnspacing=1.0)

                    # 获取图例的高度信息（用于确定p值位置）
                    legend_bbox = legend.get_window_extent()
                    legend_height = legend_bbox.height

                    # 将图例的bbox坐标转换为数据坐标
                    legend_bbox_transformed = legend_bbox.transformed(plt.gca().transAxes.inverted())
                    legend_bottom = legend_bbox_transformed.y0  # 图例底部位置

                # 2. 添加显著性检验（p值）
                p_value_text = None
                if has_lcc_data and has_non_lcc_data and len(lcc_centralities) > 10 and len(non_lcc_centralities) > 10:
                    try:
                        from scipy import stats
                        t_stat, p_value = stats.ttest_ind(lcc_centralities, non_lcc_centralities,
                                                          equal_var=False, nan_policy='omit')

                        # 格式化p值文本
                        if p_value < 0.001:
                            significance_text = f'p = {p_value:.2e}'
                        else:
                            significance_text = f'p = {p_value:.3f}'

                        if p_value < 0.05:
                            significance_text += '**' if p_value < 0.01 else '*'
                            pvalue_color = 'red'
                        else:
                            pvalue_color = 'black'

                        # 将p值放在图例下方右边
                        # 根据图例底部位置计算p值的位置
                        pvalue_y_position = 0.92  # 默认位置，在图例下方

                        # 如果有图例，调整p值位置到图例下方
                        if legend_elements:
                            # 图例底部下方一点
                            pvalue_y_position = legend_bottom - 0.05
                            # 确保不会太低
                            pvalue_y_position = max(pvalue_y_position, 0.75)

                        # 添加p值文本
                        p_value_text = plt.text(0.98,  # x位置：右边
                                                pvalue_y_position,  # y位置：图例下方
                                                significance_text,
                                                transform=plt.gca().transAxes,
                                                ha='right',  # 右对齐
                                                va='top',  # 上对齐
                                                fontsize=8,
                                                fontweight='bold' if p_value < 0.05 else 'normal',
                                                color=pvalue_color,
                                                bbox=dict(boxstyle='round,pad=0.2',
                                                          facecolor='white',
                                                          alpha=0.9,
                                                          edgecolor=pvalue_color if p_value < 0.05 else 'gray',
                                                          linewidth=1))

                    except Exception as e:
                        logger.debug(f"显著性检验失败: {str(e)}")

                # 3. 调整图形布局，为图例和p值腾出空间
                # 获取当前的y轴范围
                y_min, y_max = plt.ylim()

                # 如果有p值文本，确保图形上方有足够空间
                if p_value_text:
                    # 稍微增加顶部空间
                    plt.ylim(y_min, y_max * 1.08)
                elif legend_elements:
                    # 只有图例的情况
                    plt.ylim(y_min, y_max * 1.05)

                # 调整子图位置，防止被遮挡
                plt.subplots_adjust(top=0.85)

            else:
                # 没有数据的情况
                plt.text(0.5, 0.5, 'No centrality data available',
                         ha='center', va='center', transform=plt.gca().transAxes,
                         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                plt.title('Centrality Comparison', fontsize=10)

        # 子图12: LCC vs Non-LCC 度分布对比 (log-log)
        if use_largest_component and 'in_lcc' in centrality_df.columns and 'degree' in centrality_df.columns:
            ax = plt.subplot(3, 4, 12)  # 使用ax而不是plt，避免创建新图形

            # 分离LCC和非LCC节点的度数据
            lcc_degrees = centrality_df[centrality_df['in_lcc']]['degree'].dropna()
            non_lcc_degrees = centrality_df[~centrality_df['in_lcc']]['degree'].dropna()

            # 检查是否有数据
            has_lcc_degrees = len(lcc_degrees) > 0
            has_non_lcc_degrees = len(non_lcc_degrees) > 0

            if has_lcc_degrees or has_non_lcc_degrees:
                # 计算度分布（概率密度）
                from collections import Counter
                import numpy as np

                # 为LCC节点计算度分布
                lcc_degree_counts = Counter(lcc_degrees)
                lcc_degrees_vals = np.array(sorted(lcc_degree_counts.keys()))
                lcc_degrees_freq = np.array([lcc_degree_counts[k] for k in lcc_degrees_vals])
                lcc_degrees_prob = lcc_degrees_freq / len(lcc_degrees)

                # 为非LCC节点计算度分布
                non_lcc_degree_counts = Counter(non_lcc_degrees)
                non_lcc_degrees_vals = np.array(sorted(non_lcc_degree_counts.keys()))
                non_lcc_degrees_freq = np.array([non_lcc_degree_counts[k] for k in non_lcc_degrees_vals])
                non_lcc_degrees_prob = non_lcc_degrees_freq / len(non_lcc_degrees)

                # 绘制LCC度分布（红色）
                lcc_fit_label = None
                non_lcc_fit_label = None

                if has_lcc_degrees and len(lcc_degrees_vals) > 1:
                    ax.scatter(lcc_degrees_vals, lcc_degrees_prob,
                               color='red', s=30, alpha=0.7,
                               edgecolors='black', linewidth=0.5,
                               label=f'LCC Nodes (n={len(lcc_degrees)})',
                               zorder=3)

                    # 对LCC度分布进行幂律拟合
                    if len(lcc_degrees_vals) >= 2:
                        valid_lcc_idx = (lcc_degrees_prob > 0) & (lcc_degrees_vals > 0)
                        if np.sum(valid_lcc_idx) >= 2:
                            lcc_log_k = np.log(lcc_degrees_vals[valid_lcc_idx])
                            lcc_log_p = np.log(lcc_degrees_prob[valid_lcc_idx])

                            from scipy import stats
                            lcc_slope, lcc_intercept, lcc_r_value, lcc_p_value, lcc_std_err = \
                                stats.linregress(lcc_log_k, lcc_log_p)

                            lcc_fit_x = np.logspace(np.log10(max(1, min(lcc_degrees_vals))),
                                                    np.log10(max(lcc_degrees_vals)), 100)
                            lcc_fit_y = np.exp(lcc_intercept) * lcc_fit_x ** lcc_slope

                            ax.plot(lcc_fit_x, lcc_fit_y,
                                    color='red', linewidth=1.5, linestyle='--', alpha=0.8,
                                    label=f'LCC fit: γ={lcc_slope:.2f}, r={lcc_r_value:.2f}')

                            lcc_fit_label = f'γ={lcc_slope:.2f}, r={lcc_r_value:.2f}'

                # 绘制非LCC度分布（灰色）
                if has_non_lcc_degrees and len(non_lcc_degrees_vals) > 1:
                    ax.scatter(non_lcc_degrees_vals, non_lcc_degrees_prob,
                               color='lightgray', s=30, alpha=0.7,
                               edgecolors='black', linewidth=0.5,
                               label=f'Non-LCC Nodes (n={len(non_lcc_degrees)})',
                               zorder=2)

                    # 对非LCC度分布进行幂律拟合
                    if len(non_lcc_degrees_vals) >= 2:
                        valid_non_lcc_idx = (non_lcc_degrees_prob > 0) & (non_lcc_degrees_vals > 0)
                        if np.sum(valid_non_lcc_idx) >= 2:
                            non_lcc_log_k = np.log(non_lcc_degrees_vals[valid_non_lcc_idx])
                            non_lcc_log_p = np.log(non_lcc_degrees_prob[valid_non_lcc_idx])

                            from scipy import stats
                            non_lcc_slope, non_lcc_intercept, non_lcc_r_value, non_lcc_p_value, non_lcc_std_err = \
                                stats.linregress(non_lcc_log_k, non_lcc_log_p)

                            non_lcc_fit_x = np.logspace(np.log10(max(1, min(non_lcc_degrees_vals))),
                                                        np.log10(max(non_lcc_degrees_vals)), 100)
                            non_lcc_fit_y = np.exp(non_lcc_intercept) * non_lcc_fit_x ** non_lcc_slope

                            ax.plot(non_lcc_fit_x, non_lcc_fit_y,
                                    color='gray', linewidth=1.5, linestyle=':', alpha=0.8,
                                    label=f'Non-LCC fit: γ={non_lcc_slope:.2f}, r={non_lcc_r_value:.2f}')

                            non_lcc_fit_label = f'γ={non_lcc_slope:.2f}, r={non_lcc_r_value:.2f}'

                # 设置对数坐标轴
                ax.set_xscale('log')
                ax.set_yscale('log')

                # 设置坐标轴标签
                ax.set_xlabel('Degree (k)', fontsize=9)
                ax.set_ylabel('P(k)', fontsize=9)

                # 设置标题
                ax.set_title('Degree Distribution:\nLCC vs Non-LCC',
                             fontsize=10, fontweight='bold', pad=8)

                # 设置网格
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

                # 设置坐标轴范围
                all_degrees = []
                if has_lcc_degrees:
                    all_degrees.extend(lcc_degrees_vals)
                if has_non_lcc_degrees:
                    all_degrees.extend(non_lcc_degrees_vals)

                if all_degrees:
                    min_degree = max(1, min(all_degrees))
                    max_degree = max(all_degrees)
                    ax.set_xlim(min_degree * 0.8, max_degree * 1.2)

                # 创建图例（精简版，适合子图大小）
                handles, labels = ax.get_legend_handles_labels()

                if handles:
                    # 重新组织图例
                    scatter_handles = []
                    scatter_labels = []
                    fit_handles = []
                    fit_labels = []

                    for handle, label in zip(handles, labels):
                        if 'fit:' in label:
                            fit_handles.append(handle)
                            fit_labels.append(label)
                        else:
                            scatter_handles.append(handle)
                            scatter_labels.append(label)

                    # 如果拟合参数过多，可以简化显示
                    if len(fit_labels) > 0:
                        # 创建精简图例
                        from matplotlib.patches import Patch
                        from matplotlib.lines import Line2D

                        legend_elements = []

                        # 添加散点图例
                        if has_lcc_degrees:
                            legend_elements.append(Line2D([0], [0],
                                                          marker='o', color='w',
                                                          markerfacecolor='red',
                                                          markeredgecolor='black',
                                                          markersize=6,
                                                          label=f'LCC (n={len(lcc_degrees)})'))

                        if has_non_lcc_degrees:
                            legend_elements.append(Line2D([0], [0],
                                                          marker='o', color='w',
                                                          markerfacecolor='lightgray',
                                                          markeredgecolor='black',
                                                          markersize=6,
                                                          label=f'Non-LCC (n={len(non_lcc_degrees)})'))

                        # 添加拟合参数文本
                        fit_text = []
                        if lcc_fit_label:
                            fit_text.append(f'LCC: {lcc_fit_label}')
                        if non_lcc_fit_label:
                            fit_text.append(f'Non-LCC: {non_lcc_fit_label}')

                        if fit_text:
                            # 将拟合参数放在图例的一个单独条目中
                            legend_elements.append(Patch(facecolor='none', edgecolor='none',
                                                         label='\n'.join(fit_text)))

                        # 将图例放在图形外右上角
                        ax.legend(handles=legend_elements,
                                  loc='upper left',
                                  bbox_to_anchor=(1.02, 1.0),  # 放在图形外右侧
                                  fontsize=7,
                                  framealpha=0.9,
                                  fancybox=True,
                                  borderpad=0.8)

                        # 调整子图位置，为图例腾出空间
                        plt.subplots_adjust(right=0.85)

                # 添加统计信息到图中
                stats_text = []
                if has_lcc_degrees:
                    stats_text.append(f'LCC: k̄={np.mean(lcc_degrees):.1f}')
                if has_non_lcc_degrees:
                    stats_text.append(f'Non-LCC: k̄={np.mean(non_lcc_degrees):.1f}')

                if stats_text:
                    ax.text(0.02, 0.02, '\n'.join(stats_text),
                            transform=ax.transAxes,
                            fontsize=7,
                            verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.3))

            else:
                # 没有数据的情况
                ax.text(0.5, 0.5, 'No degree data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.set_title('Degree Distribution', fontsize=10)
                ax.grid(True, alpha=0.3)
    plt.suptitle(f'DBMS Reference Network Centrality Analysis\n'
                 f'{"Using All Nodes" if not use_largest_component else "Using All Nodes with LCC Distinction"}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存可视化结果
    visualization_path = os.path.join(output_dir, "centrality_analysis_visualization.png")
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"中心性分析可视化已保存至: {visualization_path}")

    # 10. 生成分析报告
    analysis_report = {
        'network_info': {
            'num_nodes': int(G.number_of_nodes()),
            'num_edges': int(G.number_of_edges()),
            'is_connected': bool(nx.is_connected(G)),
            'used_largest_component_for_closeness': use_largest_component,
            'centrality_based_on': 'All nodes (except closeness centrality which may use LCC)'
        },
        'largest_component_info': {},
        'centrality_statistics': {},
        'top_key_nodes': {},
        'centrality_correlation_summary': {}
    }

    # 添加最大连通子图信息
    if use_largest_component:
        analysis_report['largest_component_info'] = {
            'lcc_num_nodes': int(G_lcc.number_of_nodes()),
            'lcc_num_edges': int(G_lcc.number_of_edges()),
            'lcc_coverage_ratio': float(
                G_lcc.number_of_nodes() / G.number_of_nodes()) if G.number_of_nodes() > 0 else 0,
            'lcc_is_connected': bool(nx.is_connected(G_lcc))
        }

    # 添加中心性统计信息
    for centrality_name in centrality_results.keys():
        if centrality_name in centrality_df.columns:
            # 所有节点的统计
            all_values = centrality_df[centrality_name].dropna()

            stats_dict = {}
            if len(all_values) > 0:
                stats_dict['all_nodes'] = {
                    'mean': float(all_values.mean()),
                    'median': float(all_values.median()),
                    'std': float(all_values.std()),
                    'max': float(all_values.max()),
                    'min': float(all_values.min()),
                    'count': int(len(all_values))
                }

            if stats_dict:
                analysis_report['centrality_statistics'][centrality_name] = stats_dict

    # 添加关键节点信息（基于所有节点）
    if 'composite_centrality' in centrality_df.columns:
        top_nodes = centrality_df.nlargest(min(top_k, len(centrality_df)), 'composite_centrality')
        key_nodes_list = []

        for idx, (node_id, row) in enumerate(top_nodes.iterrows(), 1):
            node_info = {
                'rank': idx,
                'node_id': str(node_id),
                'composite_centrality': float(row['composite_centrality']) if pd.notna(
                    row['composite_centrality']) else None,
                'in_lcc': bool(row['in_lcc']) if 'in_lcc' in row else True
            }

            # 添加其他中心性指标
            for centrality_name in centrality_results.keys():
                if centrality_name in row:
                    node_info[centrality_name] = float(row[centrality_name]) if pd.notna(row[centrality_name]) else None

            # 添加节点属性
            for attr_name in ['repo_name', 'repo_id', 'degree', 'node_type']:
                if attr_name in row:
                    attr_value = row[attr_name]
                    node_info[attr_name] = str(attr_value) if pd.notna(attr_value) else None

            key_nodes_list.append(node_info)

        analysis_report['top_key_nodes'] = {
            'nodes': key_nodes_list,
            'based_on_all_nodes': True,
            'selection_criteria': 'Top nodes from all nodes by composite centrality'
        }

    # 添加中心性相关性摘要
    if not centrality_correlation.empty:
        # 计算平均相关性
        centrality_correlation_numeric = centrality_correlation.select_dtypes(include=[np.number])
        mask = np.triu(np.ones(centrality_correlation_numeric.shape), k=1).astype(bool)
        upper_triangle = centrality_correlation_numeric.values[mask]

        if len(upper_triangle) > 0:
            analysis_report['centrality_correlation_summary'] = {
                'mean_correlation': float(np.nanmean(upper_triangle)),
                'median_correlation': float(np.nanmedian(upper_triangle)),
                'min_correlation': float(np.nanmin(upper_triangle)),
                'max_correlation': float(np.nanmax(upper_triangle)),
                'based_on_all_nodes': True
            }

    # 保存分析报告
    report_path = os.path.join(output_dir, "centrality_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=4, ensure_ascii=False)

    # 11. 打印关键发现
    logger.info("=" * 70)
    logger.info("中心性分析关键发现:")
    logger.info(f"1. 网络连通性分析:")
    logger.info(f"   - 总节点数: {G.number_of_nodes()}")
    logger.info(f"   - 总边数: {G.number_of_edges()}")
    logger.info(f"   - 是否连通: {'是' if nx.is_connected(G) else '否'}")

    if use_largest_component:
        logger.info(f"   - 最大连通子图(LCC): {G_lcc.number_of_nodes()} 个节点 "
                    f"({G_lcc.number_of_nodes() / G.number_of_nodes():.2%})")
        logger.info(f"   - LCC是否连通: {'是' if nx.is_connected(G_lcc) else '否'}")
        logger.info(f"   - 中心性计算基于: 所有节点（接近中心性使用LCC）")
    else:
        logger.info(f"   - 中心性计算基于: 所有节点")

    logger.info(f"2. 中心性指标统计 (所有节点):")
    for centrality_name, stats_dict in analysis_report.get('centrality_statistics', {}).items():
        if 'all_nodes' in stats_dict:
            all_stats = stats_dict['all_nodes']
            logger.info(f"   - {centrality_name}: "
                        f"均值={all_stats['mean']:.4f}, 中位数={all_stats['median']:.4f}, "
                        f"标准差={all_stats['std']:.4f}")

    if analysis_report.get('top_key_nodes', {}).get('nodes'):
        nodes = analysis_report['top_key_nodes']['nodes']
        logger.info(f"3. 前{len(nodes)}个关键节点 (基于所有节点):")
        for i, node in enumerate(nodes[:5], 1):  # 只显示前5个
            repo_name = node.get('repo_name', node.get('node_id', '未知'))
            centrality = node.get('composite_centrality', 0)
            degree = node.get('degree', '未知')
            in_lcc = "LCC" if node.get('in_lcc') else "Non-LCC"
            logger.info(f"   {i}. {repo_name} ({in_lcc}, 综合中心性: {centrality:.4f}, 度: {degree})")

    if analysis_report.get('centrality_correlation_summary', {}).get('mean_correlation') is not None:
        mean_corr = analysis_report['centrality_correlation_summary']['mean_correlation']
        logger.info(f"4. 中心性指标相关性 (所有节点):")
        logger.info(f"   - 平均相关性: {mean_corr:.4f}")
        if mean_corr > 0.7:
            logger.info(f"   - 高度相关: 各中心性指标测量相似的节点重要性")
        elif mean_corr > 0.3:
            logger.info(f"   - 中等相关: 各中心性指标部分一致")
        else:
            logger.info(f"   - 低相关: 各中心性指标测量不同的网络属性")

    logger.info("=" * 70)
    logger.info("中心性指标分析与关键节点识别完成。")

    # 返回结果
    result = {
        'analysis_report': analysis_report,
        'centrality_data': centrality_df,
        'graph_lcc': G_lcc if use_largest_component else None
    }

    return result


def compute_clustering_coefficient(G, output_dir=None, use_largest_component=True, community_detection=True, only_dbms_repo=False):
    """
    计算图的聚类系数并进行社区检测

    参数:
    G: 网络图对象 (NetworkX Graph)
    output_dir: 输出目录路径
    use_largest_component: 是否在最大连通子图上计算聚类系数
    community_detection: 是否进行社区检测

    返回:
    dict: 包含聚类系数和社区分析结果的字典
    """
    import numpy as np

    logger.info("开始计算聚类系数与社区检测...")

    if output_dir is None:
        github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
        output_dir = os.path.join(github_osdb_data_dir, f"analysis_results/clustering_community{'_only_dbms_repo' if only_dbms_repo else ''}")

    os.makedirs(output_dir, exist_ok=True)

    # 确保图有节点
    if G.number_of_nodes() == 0:
        logger.warning("图为空，无法计算聚类系数")
        return {}

    # 获取最大连通子图（用于聚类系数计算）
    if use_largest_component and not nx.is_connected(G):
        # 获取所有连通分量
        connected_components = list(nx.connected_components(G))
        if connected_components:
            # 找到最大的连通分量
            largest_component = max(connected_components, key=len)
            G_lcc = G.subgraph(largest_component).copy()

            logger.info(f"使用最大连通子图(LCC)计算聚类系数")
            logger.info(f"   - 原始图节点数: {G.number_of_nodes()}")
            logger.info(
                f"   - LCC节点数: {G_lcc.number_of_nodes()} (覆盖率: {G_lcc.number_of_nodes() / G.number_of_nodes():.2%})")
            logger.info(f"   - LCC边数: {G_lcc.number_of_edges()}")
            G_for_clustering = G_lcc
        else:
            G_lcc = G
            G_for_clustering = G
            logger.warning("图没有连通分量，使用原图计算聚类系数")
    else:
        G_lcc = G
        G_for_clustering = G
        logger.info(f"使用完整图计算聚类系数 (节点数: {G.number_of_nodes()})")

    # 准备存储结果的字典
    clustering_results = {}

    # 1. 计算全局聚类系数（传递性）
    try:
        if G_for_clustering.number_of_edges() > 0:
            global_clustering = nx.transitivity(G_for_clustering)
            # 确保是标量
            if isinstance(global_clustering, (list, tuple, np.ndarray)):
                global_clustering = float(global_clustering[0]) if len(global_clustering) > 0 else 0.0
            else:
                global_clustering = float(global_clustering)
            clustering_results['global_clustering_coefficient'] = global_clustering
            logger.info(f"全局聚类系数（传递性）: {global_clustering:.4f}")
        else:
            global_clustering = 0.0
            clustering_results['global_clustering_coefficient'] = global_clustering
            logger.warning("图没有边，全局聚类系数为0")
    except Exception as e:
        logger.error(f"计算全局聚类系数失败: {str(e)}")
        global_clustering = 0.0

    # 2. 计算平均聚类系数
    try:
        if G_for_clustering.number_of_edges() > 0:
            avg_clustering = nx.average_clustering(G_for_clustering)
            # 处理可能的返回值类型
            if isinstance(avg_clustering, dict):
                # 如果是字典，取所有值的平均值
                avg_clustering_values = list(avg_clustering.values())
                avg_clustering = float(np.mean(avg_clustering_values)) if avg_clustering_values else 0.0
                logger.info(f"average_clustering返回字典，取平均值: {avg_clustering:.4f}")
            elif isinstance(avg_clustering, (list, tuple, np.ndarray)):
                # 如果是列表或数组，取平均值
                avg_clustering = float(np.mean(avg_clustering)) if len(avg_clustering) > 0 else 0.0
                logger.info(f"average_clustering返回列表，取平均值: {avg_clustering:.4f}")
            else:
                # 如果是标量，直接转换
                avg_clustering = float(avg_clustering)

            clustering_results['average_clustering_coefficient'] = avg_clustering
            logger.info(f"平均聚类系数: {avg_clustering:.4f}")
        else:
            avg_clustering = 0.0
            clustering_results['average_clustering_coefficient'] = avg_clustering
            logger.warning("图没有边，平均聚类系数为0")
    except Exception as e:
        logger.error(f"计算平均聚类系数失败: {str(e)}")
        avg_clustering = 0.0

    # 3. 计算局部聚类系数分布
    try:
        if G_for_clustering.number_of_edges() > 0:
            local_clustering = nx.clustering(G_for_clustering)

            # 确保local_clustering是字典
            if not isinstance(local_clustering, dict):
                # 尝试转换
                if hasattr(local_clustering, 'items'):
                    local_clustering = dict(local_clustering)
                else:
                    # 如果是其他类型，创建空字典
                    logger.warning(f"局部聚类系数不是字典类型: {type(local_clustering)}")
                    local_clustering = {}

            clustering_results['local_clustering_coefficients'] = local_clustering

            # 将局部聚类系数转换为DataFrame
            if local_clustering:
                local_clustering_df = pd.DataFrame(
                    local_clustering.items(),
                    columns=['node', 'clustering_coefficient']
                )

                # 添加节点属性信息
                node_attrs = {}
                for node in G_for_clustering.nodes():
                    attrs = G_for_clustering.nodes[node]
                    for attr_name, attr_value in attrs.items():
                        if attr_name not in node_attrs:
                            node_attrs[attr_name] = {}
                        node_attrs[attr_name][node] = attr_value

                # 将节点属性添加到DataFrame
                for attr_name, attr_dict in node_attrs.items():
                    local_clustering_df[attr_name] = local_clustering_df['node'].map(attr_dict)

                # 计算聚类系数分布统计
                clustering_values = list(local_clustering.values())
                clustering_stats = {
                    'mean': float(np.mean(clustering_values)) if clustering_values else 0.0,
                    'median': float(np.median(clustering_values)) if clustering_values else 0.0,
                    'std': float(np.std(clustering_values)) if clustering_values else 0.0,
                    'min': float(np.min(clustering_values)) if clustering_values else 0.0,
                    'max': float(np.max(clustering_values)) if clustering_values else 0.0,
                    'num_nodes_with_clustering': len([v for v in clustering_values if v > 0])
                }
                clustering_results['clustering_distribution_stats'] = clustering_stats

                logger.info(f"局部聚类系数统计 - 均值: {clustering_stats['mean']:.4f}, "
                            f"中位数: {clustering_stats['median']:.4f}, "
                            f"标准差: {clustering_stats['std']:.4f}")
            else:
                local_clustering_df = pd.DataFrame()
                clustering_stats = {
                    'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'num_nodes_with_clustering': 0
                }
        else:
            local_clustering = {}
            local_clustering_df = pd.DataFrame()
            clustering_stats = {
                'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'num_nodes_with_clustering': 0
            }
    except Exception as e:
        logger.error(f"计算局部聚类系数失败: {str(e)}")
        local_clustering = {}
        local_clustering_df = pd.DataFrame()
        clustering_stats = {}

    # 4. 计算度相关的聚类系数
    try:
        if G_for_clustering.number_of_edges() > 0 and local_clustering:
            # 创建度与聚类系数的DataFrame
            degree_clustering_data = []
            for node, coeff in local_clustering.items():
                degree = G_for_clustering.degree(node)
                degree_clustering_data.append({
                    'node': node,
                    'degree': degree,
                    'clustering_coefficient': coeff
                })

            degree_clustering_df = pd.DataFrame(degree_clustering_data)

            # 计算度分组的平均聚类系数
            if len(degree_clustering_df) > 0:
                # 创建度分组
                unique_degrees = len(set(degree_clustering_df['degree']))
                num_bins = min(10, unique_degrees)

                if num_bins > 1:
                    degree_clustering_df['degree_group'] = pd.cut(
                        degree_clustering_df['degree'],
                        bins=num_bins,
                        include_lowest=True
                    )

                    grouped_stats = degree_clustering_df.groupby('degree_group').agg({
                        'clustering_coefficient': ['mean', 'median', 'std', 'count'],
                        'degree': 'mean'
                    }).round(4)

                    grouped_stats.columns = ['_'.join(col).strip() for col in grouped_stats.columns.values]
                    clustering_results['degree_clustering_stats'] = grouped_stats.to_dict('index')

                    logger.info("计算度相关的聚类系数")
                else:
                    logger.info("度值种类太少，跳过度分组分析")
        else:
            degree_clustering_df = pd.DataFrame()
            logger.info("没有局部聚类系数数据，跳过度相关的聚类系数计算")
    except Exception as e:
        logger.warning(f"计算度相关的聚类系数失败: {str(e)}")
        degree_clustering_df = pd.DataFrame()

    # 5. 社区检测
    community_results = {}
    if community_detection and G_for_clustering.number_of_edges() > 0:
        try:
            # 尝试导入社区检测库
            try:
                import community as community_louvain
                use_louvain = True
                logger.info("使用python-louvain库进行社区检测")
            except ImportError:
                try:
                    import networkx.algorithms.community as nx_community
                    use_louvain = False
                    logger.info("使用NetworkX内置算法进行社区检测")
                except:
                    use_louvain = False
                    logger.warning("未找到社区检测库，跳过社区检测")

            if use_louvain or not use_louvain:
                # 转换为无向图进行社区检测
                if G_for_clustering.is_directed():
                    G_undirected = G_for_clustering.to_undirected()
                else:
                    G_undirected = G_for_clustering

                # 进行社区检测
                if use_louvain:
                    # 使用Louvain算法
                    partition = community_louvain.best_partition(G_undirected)
                    logger.info(f"Louvain算法检测到 {len(set(partition.values()))} 个社区")
                else:
                    # 使用NetworkX的贪心算法
                    communities_generator = nx_community.greedy_modularity_communities(G_undirected)
                    communities = list(communities_generator)
                    # 转换为节点到社区ID的映射
                    partition = {}
                    for i, comm in enumerate(communities):
                        for node in comm:
                            partition[node] = i
                    logger.info(f"贪心算法检测到 {len(communities)} 个社区")

                # 计算模块度
                if use_louvain:
                    modularity = community_louvain.modularity(partition, G_undirected)
                else:
                    # 使用NetworkX计算模块度
                    import networkx.algorithms.community.quality as nx_quality
                    # 将partition转换为社区列表格式
                    comm_dict = {}
                    for node, comm_id in partition.items():
                        if comm_id not in comm_dict:
                            comm_dict[comm_id] = set()
                        comm_dict[comm_id].add(node)
                    communities_list = list(comm_dict.values())
                    modularity = nx_quality.modularity(G_undirected, communities_list)

                # 社区统计
                community_sizes = {}
                for node, comm_id in partition.items():
                    if comm_id not in community_sizes:
                        community_sizes[comm_id] = 0
                    community_sizes[comm_id] += 1

                # 社区大小统计
                community_size_stats = {
                    'num_communities': len(community_sizes),
                    'max_community_size': max(community_sizes.values()) if community_sizes else 0,
                    'min_community_size': min(community_sizes.values()) if community_sizes else 0,
                    'avg_community_size': float(np.mean(list(community_sizes.values()))) if community_sizes else 0.0,
                    'median_community_size': float(
                        np.median(list(community_sizes.values()))) if community_sizes else 0.0,
                    'modularity': float(modularity)
                }

                # 将社区信息添加到聚类系数DataFrame
                if not local_clustering_df.empty:
                    local_clustering_df['community_id'] = local_clustering_df['node'].map(partition)
                    # 添加社区大小信息
                    local_clustering_df['community_size'] = local_clustering_df['community_id'].map(community_sizes)

                community_results = {
                    'partition': partition,
                    'community_sizes': community_sizes,
                    'community_statistics': community_size_stats,
                    'modularity': float(modularity)
                }

                logger.info(f"社区检测结果:")
                logger.info(f"  - 社区数量: {community_size_stats['num_communities']}")
                logger.info(f"  - 最大社区大小: {community_size_stats['max_community_size']}")
                logger.info(f"  - 最小社区大小: {community_size_stats['min_community_size']}")
                logger.info(f"  - 平均社区大小: {community_size_stats['avg_community_size']:.1f}")
                logger.info(f"  - 模块度: {modularity:.4f}")

        except Exception as e:
            logger.error(f"社区检测失败: {str(e)}")
            traceback.print_exc()
            community_results = {}

    # 6. 保存结果
    # 保存聚类系数数据
    if not local_clustering_df.empty:
        clustering_csv_path = os.path.join(output_dir, "clustering_coefficients.csv")
        local_clustering_df.to_csv(clustering_csv_path, index=False, encoding='utf-8')
        logger.info(f"聚类系数数据已保存至: {clustering_csv_path}")

    # 保存度相关的聚类系数数据
    if not degree_clustering_df.empty:
        degree_clustering_path = os.path.join(output_dir, "degree_clustering_analysis.csv")
        degree_clustering_df.to_csv(degree_clustering_path, index=False, encoding='utf-8')
        logger.info(f"度相关的聚类系数数据已保存至: {degree_clustering_path}")

    # 保存社区检测结果
    if community_results:
        community_json_path = os.path.join(output_dir, "community_detection_results.json")
        # 确保结果可序列化
        serializable_community_results = {}
        for key, value in community_results.items():
            if key == 'partition':
                serializable_community_results[key] = {str(k): int(v) for k, v in value.items()}
            elif key == 'community_sizes':
                serializable_community_results[key] = {int(k): int(v) for k, v in value.items()}
            elif key == 'community_statistics':
                serializable_community_results[key] = {
                    k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                    for k, v in value.items()}
            else:
                serializable_community_results[key] = value

        with open(community_json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_community_results, f, indent=4, ensure_ascii=False)
        logger.info(f"社区检测结果已保存至: {community_json_path}")

    # 7. 可视化分析
    plt.figure(figsize=(16, 12))

    # 子图1: 局部聚类系数分布
    plt.subplot(2, 3, 1)
    if local_clustering:
        clustering_values = list(local_clustering.values())
        plt.hist(clustering_values, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Local Clustering Coefficient', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Distribution of Local Clustering Coefficients', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加统计信息
        if clustering_stats:
            stats_text = f"Mean: {clustering_stats.get('mean', 0):.4f}\n"
            stats_text += f"Median: {clustering_stats.get('median', 0):.4f}\n"
            stats_text += f"Std: {clustering_stats.get('std', 0):.4f}"
            plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No clustering coefficient data',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Local Clustering Coefficient Distribution', fontsize=12)

    # 子图2: 度与聚类系数的关系（展示负相关模式）
    plt.subplot(2, 3, 2)
    if not degree_clustering_df.empty:
        # 绘制散点图
        scatter = plt.scatter(degree_clustering_df['degree'],
                              degree_clustering_df['clustering_coefficient'],
                              alpha=0.6, s=20, edgecolors='black', linewidth=0.5,
                              c=degree_clustering_df['clustering_coefficient'],
                              cmap='viridis', vmin=0, vmax=1)

        plt.xlabel('Degree (k)', fontsize=10)
        plt.ylabel('Clustering Coefficient (C)', fontsize=10)
        plt.title('Degree vs Clustering Coefficient\n(Typical C(k) ~ k^(-α) Pattern)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = plt.colorbar(scatter, shrink=0.8)
        cbar.set_label('Clustering Coefficient', fontsize=9)

        # 添加典型负相关模式的参考线（1/k曲线）
        if len(degree_clustering_df) > 1:
            try:
                # 获取数据范围
                x_min = degree_clustering_df['degree'].min()
                x_max = degree_clustering_df['degree'].max()
                y_max = degree_clustering_df['clustering_coefficient'].max()

                # 绘制典型的1/k参考曲线（负相关模式）
                x_ref = np.logspace(np.log10(max(1, x_min)), np.log10(x_max), 100)

                # 尝试不同的负幂律：C(k) ~ k^(-α)，α通常在0.5-1.5之间
                ref_lines = []
                labels = []
                colors = []

                for alpha, color, label in [(0.5, 'orange', 'k^(-0.5)'),
                                            (1.0, 'red', 'k^(-1)'),
                                            (1.5, 'purple', 'k^(-1.5)')]:
                    y_ref = y_max * np.power(x_ref, -alpha) / np.power(x_min, -alpha)
                    line, = plt.plot(x_ref, y_ref, color=color, linestyle='--',
                                     linewidth=1.5, alpha=0.7)
                    ref_lines.append(line)
                    labels.append(label)
                    colors.append(color)

                # 设置对数坐标以更好展示幂律关系
                plt.xscale('log')
                plt.yscale('log')

                # 计算相关性统计
                from scipy import stats
                pearson_corr, pearson_p = stats.pearsonr(
                    degree_clustering_df['degree'],
                    degree_clustering_df['clustering_coefficient']
                )
                spearman_corr, spearman_p = stats.spearmanr(
                    degree_clustering_df['degree'],
                    degree_clustering_df['clustering_coefficient']
                )

                # 创建自定义图例
                from matplotlib.patches import Patch
                from matplotlib.lines import Line2D

                # 创建图例元素
                legend_elements = [
                    Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5, label='k^(-0.5)'),
                    Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='k^(-1.0)'),
                    Line2D([0], [0], color='purple', linestyle='--', linewidth=1.5, label='k^(-1.5)'),
                    Patch(facecolor='white', edgecolor='black', alpha=0.8,
                          label=f'Pearson: r={pearson_corr:.3f}\nSpearman: ρ={spearman_corr:.3f}')
                ]

                # 将图例放在下方一行，靠右对齐
                legend = plt.legend(handles=legend_elements,
                                    loc='lower right',
                                    bbox_to_anchor=(0.98, 0.02),  # 右下角，稍微向内偏移
                                    fontsize=8,
                                    frameon=True,
                                    fancybox=True,
                                    framealpha=0.8,
                                    borderaxespad=0.5,
                                    ncol=2)  # 两列布局

                # 设置图例标题
                legend.set_title('Reference Curves & Correlations', prop={'size': 9, 'weight': 'bold'})

                logger.info(f"度-聚类系数相关性: Pearson r={pearson_corr:.3f}, Spearman ρ={spearman_corr:.3f}")

                # 添加负相关趋势的文本说明（放在左上角）
                plt.text(0.62, 0.98, 'Typical scaling:\nC(k) ~ k^(-α)\nα ≈ 0.5-1.5',
                         transform=plt.gca().transAxes,
                         verticalalignment='top',
                         horizontalalignment='left',
                         fontsize=9,
                         bbox=dict(boxstyle='round',
                                   facecolor='white',
                                   edgecolor='gray',
                                   alpha=0.8,
                                   pad=0.3))

            except Exception as e:
                logger.debug(f"添加参考曲线或计算相关性失败: {str(e)}")
                # 如果不成功，使用普通坐标
                plt.xscale('linear')
                plt.yscale('linear')

                # 尝试计算相关性
                try:
                    from scipy import stats
                    pearson_corr, pearson_p = stats.pearsonr(
                        degree_clustering_df['degree'],
                        degree_clustering_df['clustering_coefficient']
                    )
                    spearman_corr, spearman_p = stats.spearmanr(
                        degree_clustering_df['degree'],
                        degree_clustering_df['clustering_coefficient']
                    )

                    # 将相关性统计放在图例中
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='white', edgecolor='black', alpha=0.8,
                              label=f'Pearson r = {pearson_corr:.3f}\nSpearman ρ = {spearman_corr:.3f}')
                    ]

                    # 图例放在右下角
                    plt.legend(handles=legend_elements,
                               loc='lower right',
                               fontsize=8,
                               frameon=True,
                               fancybox=True,
                               framealpha=0.8)

                except:
                    # 添加基本的负相关说明
                    plt.text(0.02, 0.98, 'Negative correlation:\nHigh degree → Low clustering',
                             transform=plt.gca().transAxes,
                             verticalalignment='top',
                             fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    else:
        plt.text(0.5, 0.5, 'No degree-clustering data',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Degree vs Clustering Coefficient', fontsize=12)

    # 子图3: 度分组平均聚类系数
    plt.subplot(2, 3, 3)
    if 'degree_clustering_stats' in clustering_results and clustering_results['degree_clustering_stats']:
        # 提取数据
        group_stats = clustering_results['degree_clustering_stats']
        degree_groups = list(group_stats.keys())
        avg_clustering_vals = [group_stats[g].get('clustering_coefficient_mean', 0) for g in degree_groups]

        # 格式化组标签
        group_labels = []
        for group in degree_groups:
            if isinstance(group, pd.Interval):
                label = f'[{group.left:.1f}, {group.right:.1f})'
            else:
                label = str(group)
            group_labels.append(label)

        plt.bar(range(len(avg_clustering_vals)), avg_clustering_vals, alpha=0.7)
        plt.xticks(range(len(avg_clustering_vals)), group_labels, rotation=45, fontsize=8)
        plt.xlabel('Degree Group', fontsize=10)
        plt.ylabel('Average Clustering Coefficient', fontsize=10)
        plt.title('Avg Clustering Coefficient by Degree Group', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
    else:
        plt.text(0.5, 0.5, 'No degree-grouped clustering data',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Clustering by Degree Group', fontsize=12)

    # 子图4: 社区大小分布
    plt.subplot(2, 3, 4)
    if community_results and 'community_sizes' in community_results:
        community_sizes = list(community_results['community_sizes'].values())
        if community_sizes:
            plt.hist(community_sizes, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Community Size', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('Community Size Distribution', fontsize=12)
            plt.grid(True, alpha=0.3)

            # 添加统计信息
            stats = community_results.get('community_statistics', {})
            stats_text = f"Num Communities: {stats.get('num_communities', 0)}\n"
            stats_text += f"Max Size: {stats.get('max_community_size', 0)}\n"
            stats_text += f"Avg Size: {stats.get('avg_community_size', 0):.1f}\n"
            stats_text += f"Modularity: {stats.get('modularity', 0):.4f}"
            plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     fontsize=9)
        else:
            plt.text(0.5, 0.5, 'No community data',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Community Size Distribution', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No community detection performed',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Community Size Distribution', fontsize=12)

    # 子图5: 网络社区可视化（优化版，支持大规模网络）
    plt.subplot(2, 3, 5)
    if community_results and 'partition' in community_results:
        try:
            partition = community_results['partition']
            communities = set(partition.values())

            logger.info(
                f"开始网络社区可视化，共有 {len(communities)} 个社区，{G_for_clustering.number_of_nodes()} 个节点")

            # 策略选择：根据网络大小选择不同的可视化策略
            total_nodes = G_for_clustering.number_of_nodes()

            if total_nodes <= 200:
                # 小规模网络：完整可视化
                logger.info("使用完整网络可视化（小规模网络）")

                # 使用spring布局
                pos = nx.spring_layout(G_for_clustering, seed=42)

                # 为每个社区分配颜色
                colormap = plt.cm.tab20
                community_colors = {comm: colormap(i % 20) for i, comm in enumerate(communities)}

                # 计算节点大小（与节点度正相关）
                node_sizes = []
                for node in G_for_clustering.nodes():
                    degree = G_for_clustering.degree(node)
                    node_sizes.append(max(10, degree * 5))  # 最小10，与度值正相关

                # 绘制节点（按社区着色）
                for comm_id in communities:
                    nodes_in_comm = [node for node in partition if partition[node] == comm_id]
                    if nodes_in_comm:
                        # 获取这些节点的位置、大小和颜色
                        node_positions = {node: pos[node] for node in nodes_in_comm if node in pos}
                        comm_sizes = [node_sizes[list(G_for_clustering.nodes()).index(node)]
                                      for node in nodes_in_comm if node in pos]
                        comm_colors = [community_colors[comm_id]] * len(node_positions)

                        # 绘制节点
                        nx.draw_networkx_nodes(G_for_clustering, pos, nodelist=list(node_positions.keys()),
                                               node_color=comm_colors,
                                               node_size=comm_sizes, alpha=0.8)

                # 绘制边
                nx.draw_networkx_edges(G_for_clustering, pos, alpha=0.05, width=0.3, edge_color='gray')

                plt.title(f'Network with Communities\n({len(communities)} communities, {total_nodes} nodes)',
                          fontsize=12)

            elif total_nodes <= 1000:
                # 中等规模网络：提取主要社区和关键节点
                logger.info("使用主要社区和关键节点可视化（中等规模网络）")

                # 1. 识别最大的N个社区
                community_sizes = community_results.get('community_sizes', {})
                sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)

                # 取前5个最大的社区
                top_n = min(5, len(sorted_communities))
                top_communities = [comm_id for comm_id, size in sorted_communities[:top_n]]

                # 2. 从这些社区中提取度最高的k个节点
                top_k_per_community = 20  # 每个社区取前k个节点
                important_nodes = []

                for comm_id in top_communities:
                    # 获取该社区的所有节点
                    comm_nodes = [node for node, cid in partition.items() if cid == comm_id]

                    # 计算这些节点的度
                    node_degrees = [(node, G_for_clustering.degree(node)) for node in comm_nodes]

                    # 按度值排序，取前k个
                    node_degrees.sort(key=lambda x: x[1], reverse=True)
                    top_nodes = node_degrees[:top_k_per_community]

                    important_nodes.extend([node for node, _ in top_nodes])

                logger.info(f"选取了 {len(important_nodes)} 个重要节点进行可视化")

                # 3. 创建子图进行可视化
                if important_nodes:
                    # 提取包含重要节点及其邻居的子图
                    important_subgraph = G_for_clustering.subgraph(important_nodes).copy()

                    # 计算节点大小（与节点度正相关）
                    node_sizes = {}
                    for node in important_subgraph.nodes():
                        degree = important_subgraph.degree(node)
                        node_sizes[node] = max(15, degree * 8)  # 最小15，与度值正相关

                    # 使用spring布局
                    pos = nx.spring_layout(important_subgraph, seed=42, k=1 / np.sqrt(len(important_nodes)))

                    # 为每个社区分配颜色
                    colormap = plt.cm.tab20
                    community_colors = {comm: colormap(i % 20) for i, comm in enumerate(communities)}

                    # 绘制节点（按社区着色）
                    for comm_id in top_communities:
                        # 获取该社区在当前子图中的节点
                        comm_nodes_in_subgraph = [node for node in important_subgraph.nodes()
                                                  if node in partition and partition[node] == comm_id]

                        if comm_nodes_in_subgraph:
                            # 获取节点大小
                            comm_sizes = [node_sizes[node] for node in comm_nodes_in_subgraph]
                            comm_colors = [community_colors[comm_id]] * len(comm_nodes_in_subgraph)

                            # 绘制节点
                            nx.draw_networkx_nodes(important_subgraph, pos,
                                                   nodelist=comm_nodes_in_subgraph,
                                                   node_color=comm_colors,
                                                   node_size=comm_sizes, alpha=0.8)

                    # 绘制边
                    nx.draw_networkx_edges(important_subgraph, pos, alpha=0.1,
                                           width=0.5, edge_color='gray')

                    # 添加节点标签（只标注度最高的几个节点）
                    if len(important_nodes) <= 30:
                        # 为重要节点添加标签
                        labels = {}
                        for node in important_subgraph.nodes():
                            if 'repo_name' in G_for_clustering.nodes[node]:
                                repo_name = G_for_clustering.nodes[node].get('repo_name', str(node))
                                labels[node] = repo_name.split('/')[-1][:15]  # 只显示项目名，截断

                        nx.draw_networkx_labels(important_subgraph, pos, labels,
                                                font_size=6, font_weight='bold',
                                                bbox=dict(facecolor='white', alpha=0.7,
                                                          edgecolor='none', pad=1))

                    plt.title(f'Top {top_n} Communities with Key Nodes\n'
                              f'({len(important_nodes)} key nodes shown)',
                              fontsize=12)
                else:
                    # 如果没有重要节点，显示统计信息
                    plt.text(0.5, 0.5, f'Large Network: {total_nodes} nodes\n'
                                       f'Too many nodes for detailed visualization\n'
                                       f'Showing community statistics instead',
                             ha='center', va='center', transform=plt.gca().transAxes,
                             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    plt.title('Network Community Structure', fontsize=12)

            else:
                # 超大规模网络：显示社区统计和弦图
                logger.info("使用社区统计和弦图可视化（超大规模网络）")

                # 获取社区大小信息
                community_sizes = community_results.get('community_sizes', {})

                if community_sizes:
                    # 创建社区大小条形图
                    sorted_sizes = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)

                    # 取前10个最大的社区
                    top_n = min(10, len(sorted_sizes))
                    top_communities = sorted_sizes[:top_n]

                    # 提取社区ID和大小
                    comm_ids = [f'Comm{i + 1}' for i in range(top_n)]
                    comm_sizes = [size for _, size in top_communities]

                    # 创建条形图
                    bars = plt.bar(range(top_n), comm_sizes, alpha=0.7,
                                   color=plt.cm.tab20(range(top_n)))

                    plt.xticks(range(top_n), comm_ids, rotation=45, fontsize=8)
                    plt.xlabel('Community', fontsize=10)
                    plt.ylabel('Number of Nodes', fontsize=10)

                    # 添加数值标签
                    for i, (rect, size) in enumerate(zip(bars, comm_sizes)):
                        height = rect.get_height()
                        plt.text(rect.get_x() + rect.get_width() / 2., height + max(comm_sizes) * 0.01,
                                 f'{size}', ha='center', va='bottom', fontsize=8)

                    # 添加其他社区的信息
                    if len(sorted_sizes) > top_n:
                        other_size = sum(size for _, size in sorted_sizes[top_n:])
                        plt.text(0.02, 0.98,
                                 f'Other {len(sorted_sizes) - top_n} communities:\n'
                                 f'{other_size} nodes total',
                                 transform=plt.gca().transAxes,
                                 verticalalignment='top', fontsize=8,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    plt.title(f'Top {top_n} Largest Communities\n'
                              f'(Total: {total_nodes} nodes, {len(community_sizes)} communities)',
                              fontsize=12)
                    plt.grid(True, alpha=0.3, axis='y')
                else:
                    plt.text(0.5, 0.5, f'Very Large Network: {total_nodes} nodes\n'
                                       f'Community visualization optimized for large networks',
                             ha='center', va='center', transform=plt.gca().transAxes,
                             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    plt.title('Large Network Community Structure', fontsize=12)

            plt.axis('off' if total_nodes <= 1000 else 'on')

        except Exception as e:
            logger.warning(f"网络社区可视化失败: {str(e)}")
            plt.text(0.5, 0.5, 'Community visualization failed\n'
                               f'Error: {str(e)[:50]}...',
                     ha='center', va='center', transform=plt.gca().transAxes,
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.title('Network with Communities', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No community data for visualization',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title('Network with Communities', fontsize=12)

    # 子图6: 聚类系数与社区大小的关系
    plt.subplot(2, 3, 6)
    if community_results and not local_clustering_df.empty and 'community_id' in local_clustering_df.columns:
        # 计算每个社区的平均聚类系数
        community_clustering = local_clustering_df.groupby('community_id').agg({
            'clustering_coefficient': 'mean',
            'community_size': 'first'
        }).reset_index()

        if len(community_clustering) > 1:
            plt.scatter(community_clustering['community_size'],
                        community_clustering['clustering_coefficient'],
                        alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            plt.xlabel('Community Size', fontsize=10)
            plt.ylabel('Average Clustering Coefficient', fontsize=10)
            plt.title('Community Size vs Avg Clustering Coefficient', fontsize=12)
            plt.grid(True, alpha=0.3)

            # 添加回归线
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    community_clustering['community_size'],
                    community_clustering['clustering_coefficient']
                )
                x_range = np.linspace(community_clustering['community_size'].min(),
                                      community_clustering['community_size'].max(), 100)
                y_pred = slope * x_range + intercept
                plt.plot(x_range, y_pred, 'r-', linewidth=2,
                         label=f'r={r_value:.3f}, p={p_value:.3e}')
                plt.legend(fontsize=9)
            except:
                pass
        else:
            plt.text(0.5, 0.5, 'Insufficient community data',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Community Clustering Analysis', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No community-clustering data',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Community Clustering Analysis', fontsize=12)

    plt.suptitle(f'Clustering Coefficient and Community Structure Analysis\n'
                 f'{"Based on Largest Connected Component" if use_largest_component else "Based on Complete Graph"}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存可视化结果
    visualization_path = os.path.join(output_dir, "clustering_community_analysis.png")
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"聚类系数与社区结构分析可视化已保存至: {visualization_path}")

    # 8. 生成分析报告
    analysis_report = {
        'network_info': {
            'num_nodes': int(G_for_clustering.number_of_nodes()),
            'num_edges': int(G_for_clustering.number_of_edges()),
            'is_connected': bool(nx.is_connected(G_for_clustering)),
            'based_on_lcc': use_largest_component
        },
        'clustering_coefficient_analysis': {
            'global_clustering_coefficient': float(global_clustering),
            'average_clustering_coefficient': float(avg_clustering),
            'local_clustering_distribution': clustering_stats if clustering_stats else {}
        },
        'community_analysis': community_results.get('community_statistics', {}) if community_results else {}
    }

    # 如果进行了社区检测，添加详细信息
    if community_results:
        analysis_report['community_analysis'].update({
            'modularity': float(community_results.get('modularity', 0)),
            'num_communities': community_results.get('community_statistics', {}).get('num_communities', 0)
        })

    # 保存分析报告
    report_path = os.path.join(output_dir, "clustering_community_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=4, ensure_ascii=False)

    # 9. 打印关键发现
    logger.info("=" * 70)
    logger.info("聚类系数与社区结构分析关键发现:")
    logger.info(f"1. 网络基本信息:")
    logger.info(f"   - 计算基于: {'最大连通子图(LCC)' if use_largest_component else '完整图'}")
    logger.info(f"   - 节点数: {G_for_clustering.number_of_nodes()}")
    logger.info(f"   - 边数: {G_for_clustering.number_of_edges()}")
    logger.info(f"   - 是否连通: {'是' if nx.is_connected(G_for_clustering) else '否'}")

    logger.info(f"2. 聚类系数分析:")
    logger.info(f"   - 全局聚类系数（传递性）: {global_clustering:.4f}")
    logger.info(f"   - 平均聚类系数: {avg_clustering:.4f}")

    if clustering_stats:
        logger.info(f"   - 局部聚类系数统计:")
        logger.info(f"     * 均值: {clustering_stats.get('mean', 0):.4f}")
        logger.info(f"     * 中位数: {clustering_stats.get('median', 0):.4f}")
        logger.info(f"     * 标准差: {clustering_stats.get('std', 0):.4f}")
        logger.info(f"     * 最小值: {clustering_stats.get('min', 0):.4f}")
        logger.info(f"     * 最大值: {clustering_stats.get('max', 0):.4f}")

    if community_results:
        comm_stats = community_results.get('community_statistics', {})
        logger.info(f"3. 社区检测结果:")
        logger.info(f"   - 社区数量: {comm_stats.get('num_communities', 0)}")
        logger.info(f"   - 模块度: {comm_stats.get('modularity', 0):.4f}")
        logger.info(f"   - 最大社区大小: {comm_stats.get('max_community_size', 0)}")
        logger.info(f"   - 最小社区大小: {comm_stats.get('min_community_size', 0)}")
        logger.info(f"   - 平均社区大小: {comm_stats.get('avg_community_size', 0):.1f}")

        # 模块度解读
        modularity = comm_stats.get('modularity', 0)
        if modularity > 0.7:
            logger.info(f"   - 模块度解读: 非常强的社区结构")
        elif modularity > 0.3:
            logger.info(f"   - 模块度解读: 中等强度的社区结构")
        else:
            logger.info(f"   - 模块度解读: 较弱的社区结构")

    logger.info("=" * 70)
    logger.info("聚类系数与社区结构分析完成。")

    # 返回结果
    result = {
        'analysis_report': analysis_report,
        'clustering_data': local_clustering_df if not local_clustering_df.empty else None,
        'community_results': community_results if community_results else None,
        'graph_for_analysis': G_for_clustering
    }

    return result


def compare_subdomains_referenced_type(split_dfs_by_category_label, df_dbms_repos_ref_node_agg_dict,
                                       output_dir=None, tar_type_col="tar_entity_type_fine_grained"):
    """
    比较不同子领域（category_label）的引用类型分布差异

    参数:
    split_dfs_by_category_label: 按category_label划分的DataFrame字典 {label: df, ...}
    df_dbms_repos_ref_node_agg_dict: 引用关系字典 {repo_key: df_ref, ...}
    output_dir: 输出目录路径
    tar_type_col: 被引用实体类型的列名

    返回:
    dict: 包含子领域引用类型差异分析结果的字典
    """
    import numpy as np
    from scipy import stats

    logger.info("开始分析不同子领域的引用类型分布差异...")

    if output_dir is None:
        github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
        output_dir = os.path.join(github_osdb_data_dir, "analysis_results/subdomain_reference_analysis")

    os.makedirs(output_dir, exist_ok=True)

    # 确保输入数据有效
    if not split_dfs_by_category_label or not df_dbms_repos_ref_node_agg_dict:
        logger.warning("输入数据为空，无法进行分析")
        return {}

    # 1. 准备数据：将引用数据按子领域分类
    subdomain_ref_data = {}

    # 获取repo_name到category_label的映射
    repo_to_category = {}
    for category_label, df_category in split_dfs_by_category_label.items():
        for _, row in df_category.iterrows():
            repo_name = row['repo_name']
            if pd.notna(repo_name):
                repo_to_category[repo_name] = category_label

    logger.info(f"共映射了 {len(repo_to_category)} 个仓库到子领域")

    # 2. 按子领域统计引用类型分布
    for repo_key, df_ref in df_dbms_repos_ref_node_agg_dict.items():
        # 从repo_key提取repo_name
        # repo_key格式通常为: org_repo_2023.csv 或类似
        repo_name = None
        for known_repo_name in repo_to_category.keys():
            if known_repo_name.replace('/', '_') in repo_key:
                repo_name = known_repo_name
                break

        if repo_name and repo_name in repo_to_category:
            category_label = repo_to_category[repo_name]

            if category_label not in subdomain_ref_data:
                subdomain_ref_data[category_label] = {
                    'repo_count': 0,
                    'ref_counts': pd.Series(dtype=int),
                    'repo_names': []
                }

            # 统计该仓库的引用类型
            if tar_type_col in df_ref.columns:
                valid_refs = df_ref[df_ref[tar_type_col].notna()]
                if len(valid_refs) > 0:
                    ref_counts = valid_refs[tar_type_col].value_counts()

                    if subdomain_ref_data[category_label]['ref_counts'].empty:
                        subdomain_ref_data[category_label]['ref_counts'] = ref_counts
                    else:
                        # 合并统计
                        subdomain_ref_data[category_label]['ref_counts'] = (
                            subdomain_ref_data[category_label]['ref_counts']
                            .add(ref_counts, fill_value=0)
                            .astype(int)
                        )

                    subdomain_ref_data[category_label]['repo_count'] += 1
                    subdomain_ref_data[category_label]['repo_names'].append(repo_name)

    logger.info(f"成功统计了 {len(subdomain_ref_data)} 个子领域的引用数据")

    # 3. 创建对比分析DataFrame
    all_ref_types = set()
    for category_data in subdomain_ref_data.values():
        all_ref_types.update(category_data['ref_counts'].index.tolist())

    all_ref_types = sorted(list(all_ref_types))

    # 创建对比表格
    comparison_data = []
    for category_label, category_data in subdomain_ref_data.items():
        row = {'category_label': category_label, 'repo_count': category_data['repo_count']}

        # 添加各引用类型的计数
        for ref_type in all_ref_types:
            row[ref_type] = category_data['ref_counts'].get(ref_type, 0)

        # 计算总数和比例
        total_refs = sum(category_data['ref_counts'].values)
        row['total_refs'] = total_refs

        # 计算主要引用类型的比例
        if total_refs > 0:
            for ref_type in all_ref_types[:5]:  # 前5种引用类型
                if ref_type in row:
                    row[f'{ref_type}_prop'] = row[ref_type] / total_refs

        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # 按总引用数排序
    df_comparison = df_comparison.sort_values('total_refs', ascending=False).reset_index(drop=True)

    # 4. 卡方检验：检验不同子领域引用类型分布是否有显著差异
    chi2_results = {}
    if len(subdomain_ref_data) >= 2 and len(all_ref_types) >= 2:
        try:
            # 构建列联表
            contingency_table = []
            for category_label, category_data in subdomain_ref_data.items():
                row_counts = []
                for ref_type in all_ref_types:
                    row_counts.append(category_data['ref_counts'].get(ref_type, 0))
                contingency_table.append(row_counts)

            contingency_table = np.array(contingency_table)

            # 执行卡方检验
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            chi2_results = {
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': bool(p_value < 0.05),  # 使用bool()转换
                'effect_size_cramers_v': float(np.sqrt(chi2_stat / (
                        np.sum(contingency_table) * min(contingency_table.shape[0] - 1,
                                                        contingency_table.shape[1] - 1))))
            }

            logger.info(f"卡方检验结果: χ²={chi2_stat:.4f}, df={dof}, p={p_value:.4e}")
            logger.info(f"效应量 (Cramer's V): {chi2_results['effect_size_cramers_v']:.4f}")

            if p_value < 0.05:
                logger.info("不同子领域的引用类型分布存在显著差异 (p < 0.05)")
            else:
                logger.info("不同子领域的引用类型分布无显著差异")

        except Exception as e:
            logger.error(f"卡方检验失败: {str(e)}")
            chi2_results = {'error': str(e)}
    else:
        logger.warning("子领域或引用类型数量不足，无法进行卡方检验")

    # 5. 优化后的可视化分析 - 修复布局问题
    plt.figure(figsize=(24, 18))

    # 子图1: 各子领域引用类型分布的堆叠条形图（全宽度）
    ax1 = plt.subplot(3, 2, (1, 2))  # 占用第一行的两列

    if len(subdomain_ref_data) > 0 and len(all_ref_types) > 0:
        # 选择前8种最常见的引用类型
        total_counts = pd.Series(0, index=all_ref_types)
        for category_data in subdomain_ref_data.values():
            total_counts = total_counts.add(category_data['ref_counts'], fill_value=0)

        top_ref_types = total_counts.nlargest(min(8, len(total_counts))).index.tolist()

        # 准备堆叠条形图数据
        categories = list(subdomain_ref_data.keys())
        x_pos = np.arange(len(categories))
        bottom = np.zeros(len(categories))

        # 为每种引用类型创建颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_ref_types)))

        # 绘制堆叠条形图
        bars_list = []
        for i, (ref_type, color) in enumerate(zip(top_ref_types, colors)):
            proportions = []
            for category_label in categories:
                ref_counts = subdomain_ref_data[category_label]['ref_counts']
                total = ref_counts.sum()
                if total > 0:
                    count = ref_counts.get(ref_type, 0)
                    proportions.append(count / total)
                else:
                    proportions.append(0)

            bars = ax1.bar(x_pos, proportions, bottom=bottom,
                           width=0.7, color=color, alpha=0.8,
                           label=ref_type[:15] + '...' if len(ref_type) > 15 else ref_type)
            bars_list.append(bars)
            bottom += np.array(proportions)

        ax1.set_xlabel('Subdomain (Category Label)', fontsize=12)
        ax1.set_ylabel('Proportion of Reference Types', fontsize=12)
        # 修复：恢复原来的标题位置，但增加一些上边距
        ax1.set_title('Reference Type Composition by Subdomain\n(Stacked Bar Chart)',
                      fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(x_pos)

        # 缩短子领域标签，避免重叠
        category_labels_short = []
        for cat in categories:
            if len(cat) > 15:
                # 尝试分割长标签
                if '_' in cat:
                    parts = cat.split('_')
                    if len(parts) > 2:
                        category_labels_short.append(parts[0] + '...' + parts[-1])
                    else:
                        category_labels_short.append(cat[:12] + '...')
                else:
                    category_labels_short.append(cat[:12] + '...')
            else:
                category_labels_short.append(cat)

        ax1.set_xticklabels(category_labels_short, rotation=45, ha='right', fontsize=10)
        ax1.set_yticks(np.arange(0, 1.1, 0.2))
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加图例（放在外部，避免重叠）
        ax1.legend(title='Reference Types', bbox_to_anchor=(1.02, 1),
                   loc='upper left', fontsize=9, ncol=1)

        # 修复：恢复原来的数值标签位置（被框线横穿的样子）
        for i, category in enumerate(categories):
            ref_counts = subdomain_ref_data[category]['ref_counts']
            total_refs = ref_counts.sum()
            # 恢复原来的位置（1.02）和样式
            ax1.text(i, 1.02, f'n={int(total_refs)}',
                     ha='center', va='bottom', fontsize=9, rotation=0)
    else:
        ax1.text(0.5, 0.5, 'No subdomain reference data available',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Reference Type Composition by Subdomain', fontsize=14)

    # 子图2: 雷达图（修复图例位置问题）
    ax2 = plt.subplot(3, 2, 3, polar=True)

    if len(subdomain_ref_data) > 1 and len(all_ref_types) > 0:
        # 选择前5种最常见的引用类型
        total_counts = pd.Series(0, index=all_ref_types)
        for category_data in subdomain_ref_data.values():
            total_counts = total_counts.add(category_data['ref_counts'], fill_value=0)

        top_ref_types = total_counts.nlargest(min(5, len(total_counts))).index.tolist()

        # 准备雷达图数据
        categories = list(subdomain_ref_data.keys())
        num_vars = len(top_ref_types)

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 为每个子领域创建雷达图
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

        lines = []
        labels = []
        for i, (category, color) in enumerate(zip(categories, colors)):
            # 计算该子领域各引用类型的比例
            ref_counts = subdomain_ref_data[category]['ref_counts']
            values = []
            for ref_type in top_ref_types:
                total = ref_counts.sum()
                if total > 0:
                    count = ref_counts.get(ref_type, 0)
                    values.append(count / total)
                else:
                    values.append(0)

            # 闭合数据
            values += values[:1]

            # 绘制雷达图
            line, = ax2.plot(angles, values, 'o-', linewidth=2, color=color,
                             markersize=6, label=f'{category}')
            ax2.fill(angles, values, alpha=0.15, color=color)
            lines.append(line)
            labels.append(f'{category} (n={int(ref_counts.sum())})')

        # 设置雷达图标签
        ax2.set_xticks(angles[:-1])
        # 缩短标签文本，避免重叠
        xtick_labels = []
        for t in top_ref_types:
            if len(t) > 8:
                # 对于长标签，分成两行
                words = t.split('_')
                if len(words) > 1:
                    label = '\n'.join(words[:2])
                else:
                    label = t[:4] + '...'
            else:
                label = t
            xtick_labels.append(label)

        ax2.set_xticklabels(xtick_labels, fontsize=9, ha='center')
        ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
        ax2.set_ylim(0, 1)
        ax2.set_title('Top 5 Reference Types by Subdomain\n(Radar Chart)',
                      fontsize=13, fontweight='bold', pad=20)  # 增加pad确保标题不重叠

        # 修复：先调整雷达图的大小和位置，为图例腾出空间
        # 获取当前雷达图的位置
        pos = ax2.get_position()

        # 缩小雷达图本身，为右侧图例腾出空间
        # 将宽度减少20%，向右移动一点
        ax2.set_position([pos.x0 * 1.05, pos.y0, pos.width * 0.8, pos.height])

        # 创建图例放在右侧 - 使用更紧凑的布局
        # 方法1：使用紧凑图例
        legend = ax2.legend(lines, labels,
                            loc='upper left',  # 放在左上角
                            bbox_to_anchor=(1.05, 1.0),  # 向右移动一点
                            fontsize=8,
                            title='Subdomains',
                            title_fontsize=9,
                            frameon=True,
                            fancybox=True,
                            framealpha=0.9,
                            edgecolor='black',
                            ncol=1,
                            handlelength=1.5,  # 缩短图例句柄
                            handletextpad=0.5,  # 减少文本间距
                            borderaxespad=0.5)  # 减少边框间距

        # 方法2：如果图例仍然太大，可以创建两列图例
        if len(categories) > 5:  # 如果子领域太多
            legend = ax2.legend(lines, labels,
                                loc='upper left',
                                bbox_to_anchor=(1.05, 1.0),
                                fontsize=7,  # 进一步减小字体
                                title='Subdomains',
                                title_fontsize=8,
                                frameon=True,
                                fancybox=True,
                                framealpha=0.9,
                                edgecolor='black',
                                ncol=1,  # 使用两列
                                columnspacing=0.5,  # 减少列间距
                                handlelength=1.2,
                                handletextpad=0.3)

        plt.subplots_adjust(right=0.85)  # 为右侧图例腾出空间
    else:
        # 如果不是雷达图坐标，创建普通图表
        ax2.text(0.5, 0.5, 'Insufficient data for radar chart\nNeed at least 2 subdomains',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=11)
        ax2.set_title('Reference Type Comparison', fontsize=13)

    # 子图3: 简化流向图（分组条形图）
    ax3 = plt.subplot(3, 2, 4)

    if len(subdomain_ref_data) > 0 and len(all_ref_types) > 0:
        # 按总引用数排序，选取前3个子领域
        subdomain_totals = {}
        for category_label, category_data in subdomain_ref_data.items():
            subdomain_totals[category_label] = category_data['ref_counts'].sum()

        # 按总引用数排序，选取前3个
        sorted_categories = sorted(subdomain_totals.items(), key=lambda x: x[1], reverse=True)
        categories_flow = [cat for cat, _ in sorted_categories[:min(3, len(sorted_categories))]]

        # 记录排序结果到日志
        logger.info(f"子图3 - 子领域总引用数排序（选取前3个）:")
        for i, (cat, total) in enumerate(sorted_categories[:min(5, len(sorted_categories))], 1):
            logger.info(f"  {i}. {cat}: {total} 条引用")

        total_counts = pd.Series(0, index=all_ref_types)
        for category_data in subdomain_ref_data.values():
            total_counts = total_counts.add(category_data['ref_counts'], fill_value=0)

        top_ref_types_flow = total_counts.nlargest(min(4, len(total_counts))).index.tolist()

        # 准备流向图数据
        flow_data = []
        for category in categories_flow:
            for ref_type in top_ref_types_flow:
                count = subdomain_ref_data[category]['ref_counts'].get(ref_type, 0)
                if count > 0:
                    flow_data.append({
                        'category': category,
                        'ref_type': ref_type,
                        'count': count
                    })

        if flow_data:
            df_flow = pd.DataFrame(flow_data)

            # 创建分组条形图展示流向
            bar_width = 0.25
            x_pos = np.arange(len(top_ref_types_flow))

            for i, category in enumerate(categories_flow):
                category_counts = []
                for ref_type in top_ref_types_flow:
                    count = df_flow[(df_flow['category'] == category) &
                                    (df_flow['ref_type'] == ref_type)]['count'].sum()
                    category_counts.append(count)

                offset = (i - len(categories_flow) / 2 + 0.5) * bar_width
                bars = ax3.bar(x_pos + offset, category_counts, bar_width,
                               alpha=0.7, label=category)

                # 在条形上添加数值
                for j, count in enumerate(category_counts):
                    if count > 0:
                        ax3.text(x_pos[j] + offset, count, str(int(count)),
                                 ha='center', va='bottom', fontsize=8)

            ax3.set_xlabel('Reference Types', fontsize=11)
            ax3.set_ylabel('Reference Count', fontsize=11)
            ax3.set_title('Reference Flow by Top 3 Subdomains\n(Simplified Flow Diagram)',
                          fontsize=13, fontweight='bold')

            # 设置x轴标签，缩短长标签
            xtick_labels = []
            for t in top_ref_types_flow:
                if len(t) > 12:
                    words = t.split('_')
                    if len(words) > 2:
                        xtick_labels.append(words[0][:3] + '...' + words[-1][:3])
                    else:
                        xtick_labels.append(t[:8] + '...')
                else:
                    xtick_labels.append(t)

            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=9)
            ax3.legend(title='Subdomain', fontsize=9, loc='upper right')
            ax3.grid(True, alpha=0.3, axis='y')

            # 添加卡方检验结果
            if 'chi2_statistic' in chi2_results:
                chi2_text = f"χ²={chi2_results['chi2_statistic']:.1f}"
                if chi2_results.get('significant', False):
                    chi2_text += "* (p<0.05)"
                ax3.text(0.02, 0.98, chi2_text,
                         transform=ax3.transAxes,
                         verticalalignment='top',
                         fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'Insufficient flow data',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=11)
            ax3.set_title('Reference Flow by Subdomain', fontsize=13)
    else:
        ax3.text(0.5, 0.5, 'No flow data available',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title('Reference Flow by Subdomain', fontsize=13)

    # 子图4: 多样性散点图
    ax4 = plt.subplot(3, 2, 5)

    if len(subdomain_ref_data) >= 3:
        # 计算每个子领域的引用类型多样性（香农熵）和仓库数量
        diversities = []
        repo_counts = []
        categories_diversity = []

        for category_label, category_data in subdomain_ref_data.items():
            ref_counts = category_data['ref_counts']
            total = ref_counts.sum()

            if total > 0 and len(ref_counts) > 1:
                # 计算香农熵
                proportions = ref_counts / total
                entropy = -np.sum(proportions * np.log2(proportions))
                normalized_entropy = entropy / np.log2(len(ref_counts))

                diversities.append(float(normalized_entropy))
                repo_counts.append(category_data['repo_count'])
                categories_diversity.append(category_label)

        if len(diversities) >= 3:
            # 创建散点图
            scatter = ax4.scatter(repo_counts, diversities,
                                  s=[c * 30 for c in repo_counts],  # 点大小与仓库数量成正比
                                  alpha=0.7, edgecolors='black', linewidth=0.5,
                                  c=diversities, cmap='viridis')  # 颜色与多样性相关

            ax4.set_xlabel('Number of Repositories', fontsize=11)
            ax4.set_ylabel('Reference Type Diversity\n(Normalized Shannon Entropy)', fontsize=11)
            ax4.set_title('Diversity vs Repository Count by Subdomain',
                          fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # 添加子领域标签
            for i, (repo_count, diversity, category) in enumerate(zip(repo_counts, diversities, categories_diversity)):
                ax4.annotate(category,
                             xy=(repo_count, diversity),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8,
                             alpha=0.8)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
            cbar.set_label('Diversity Index', fontsize=10)

            # 添加回归线
            if len(repo_counts) > 1:
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(repo_counts, diversities)
                    x_range = np.array([min(repo_counts), max(repo_counts)])
                    y_pred = slope * x_range + intercept
                    ax4.plot(x_range, y_pred, 'r--', linewidth=2,
                             label=f'Fit: r={r_value:.3f}')
                    ax4.legend(fontsize=9)
                except:
                    pass
        else:
            ax4.text(0.5, 0.5, 'Insufficient diversity data\n(need ≥3 subdomains)',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=11)
            ax4.set_title('Diversity vs Repository Count', fontsize=13)
    else:
        ax4.text(0.5, 0.5, 'Insufficient subdomains for diversity analysis\n(need ≥3 subdomains)',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.set_title('Diversity Analysis', fontsize=13)

    # 子图5: 帕累托图
    ax5 = plt.subplot(3, 2, 6)

    if len(subdomain_ref_data) > 0:
        # 选择引用数量最多的子领域
        max_ref_category = None
        max_ref_total = 0

        for category_label, category_data in subdomain_ref_data.items():
            ref_counts = category_data['ref_counts']
            ref_total = ref_counts.sum()

            if ref_total > max_ref_total:
                max_ref_total = ref_total
                max_ref_category = (category_label, category_data)

        if max_ref_category:
            category_label = max_ref_category[0]
            category_data = max_ref_category[1]

            ref_counts = category_data['ref_counts']
            if len(ref_counts) > 0:
                # 排序引用类型，只显示前8种
                sorted_counts = ref_counts.sort_values(ascending=False)
                display_counts = sorted_counts.head(min(8, len(sorted_counts)))

                # 计算累积百分比
                cumulative_sum = display_counts.cumsum()
                total = display_counts.sum()
                cumulative_percentage = cumulative_sum / total * 100

                # 创建条形图
                x_pos = np.arange(len(display_counts))
                bars = ax5.bar(x_pos, display_counts.values,
                               color='skyblue', alpha=0.7)

                ax5.set_xlabel('Reference Types (Top 8)', fontsize=11)
                ax5.set_ylabel('Count', fontsize=11)
                ax5.set_title(f'Top Reference Types: {category_label}\n(Total: {int(total)} references)',
                              fontsize=13, fontweight='bold')

                # 设置x轴标签
                ax5.set_xticks(x_pos)

                # 创建标签，对长标签进行处理
                xtick_labels = []
                for t in display_counts.index:
                    if len(t) > 10:
                        if '_' in t:
                            parts = t.split('_')
                            if len(parts) > 1:
                                label = parts[0] + '\n' + parts[1]
                            else:
                                label = t[:8] + '...'
                        else:
                            label = t[:8] + '...'
                    else:
                        label = t
                    xtick_labels.append(label)

                ax5.set_xticklabels(xtick_labels,
                                    rotation=45,
                                    ha='right',
                                    fontsize=9,
                                    rotation_mode='anchor')

                ax5.grid(True, alpha=0.3, axis='y')

                # 在条形上添加数值
                max_height = max(display_counts.values)
                for i, (bar, count) in enumerate(zip(bars, display_counts.values)):
                    height = bar.get_height()
                    if height > max_height * 0.1:
                        ax5.text(bar.get_x() + bar.get_width() / 2., height,
                                 f'{int(count)}', ha='center', va='bottom', fontsize=8)

                # 添加累积百分比线（次坐标轴）
                ax5_twin = ax5.twinx()
                ax5_twin.plot(x_pos, cumulative_percentage.values,
                              color='red', marker='o', linewidth=2, markersize=4)
                ax5_twin.set_ylabel('Cumulative %', fontsize=11, color='red')
                ax5_twin.tick_params(axis='y', labelcolor='red')
                ax5_twin.set_ylim(0, 110)

                # 添加80%线
                ax5_twin.axhline(y=80, color='green', linestyle='--', alpha=0.7, linewidth=1)
                ax5_twin.text(0.5, 82, '80%', color='green', fontsize=9,
                              ha='center', transform=ax5_twin.get_yaxis_transform())

                # 添加图例
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='skyblue', lw=4, label='Count'),
                    Line2D([0], [0], color='red', marker='o', lw=2, label='Cumulative %'),
                    Line2D([0], [0], color='green', linestyle='--', lw=1, label='80% line')
                ]
                ax5_twin.legend(handles=legend_elements, loc='upper left', fontsize=8)
            else:
                ax5.text(0.5, 0.5, 'No reference data for this subdomain',
                         ha='center', va='center', transform=ax5.transAxes, fontsize=11)
                ax5.set_title('Pareto Analysis', fontsize=13)
        else:
            ax5.text(0.5, 0.5, 'No subdomain with reference data',
                     ha='center', va='center', transform=ax5.transAxes, fontsize=11)
            ax5.set_title('Pareto Analysis', fontsize=13)
    else:
        ax5.text(0.5, 0.5, 'No subdomain data available',
                 ha='center', va='center', transform=ax5.transAxes, fontsize=11)
        ax5.set_title('Pareto Analysis', fontsize=13)

    plt.suptitle('Subdomain Reference Pattern Analysis\n(DBMS Projects by Category Label)',
                 fontsize=16, fontweight='bold')

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    # 保存可视化结果
    visualization_path = os.path.join(output_dir, "subdomain_reference_analysis.png")
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"子领域引用类型分析可视化已保存至: {visualization_path}")

    # 6. 保存详细数据
    comparison_csv_path = os.path.join(output_dir, "subdomain_reference_comparison.csv")
    df_comparison.to_csv(comparison_csv_path, index=False, encoding='utf-8')
    logger.info(f"子领域引用对比数据已保存至: {comparison_csv_path}")

    if chi2_results:
        chi2_path = os.path.join(output_dir, "chi2_test_results.json")
        with open(chi2_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(chi2_results), f, indent=4, ensure_ascii=False)
        logger.info(f"卡方检验结果已保存至: {chi2_path}")

    # 7. 生成分析报告
    analysis_report = {
        'subdomain_summary': {
            'num_subdomains': len(subdomain_ref_data),
            'subdomains_analyzed': list(subdomain_ref_data.keys()),
            'total_repositories_analyzed': sum(data['repo_count'] for data in subdomain_ref_data.values()),
            'total_references_analyzed': sum(sum(data['ref_counts'].values) for data in subdomain_ref_data.values())
        },
        'reference_type_summary': {
            'total_reference_types': len(all_ref_types),
            'reference_types_analyzed': all_ref_types[:20]
        },
        'chi2_test_results': convert_numpy_types(chi2_results) if chi2_results else {},
        'top_reference_types_by_subdomain': {}
    }

    for category_label, category_data in subdomain_ref_data.items():
        ref_counts = category_data['ref_counts']
        if len(ref_counts) > 0:
            top_types = ref_counts.nlargest(5)
            analysis_report['top_reference_types_by_subdomain'][category_label] = {
                'top_types': {k: int(v) for k, v in top_types.to_dict().items()},
                'repo_count': int(category_data['repo_count']),
                'total_references': int(sum(ref_counts.values))
            }

    report_path = os.path.join(output_dir, "subdomain_reference_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_report), f, indent=4, ensure_ascii=False)

    # 8. 打印关键发现
    logger.info("=" * 70)
    logger.info("子领域引用类型差异分析关键发现:")
    logger.info(f"1. 分析概况:")
    logger.info(f"   - 分析子领域数量: {len(subdomain_ref_data)}")
    logger.info(f"   - 涉及仓库总数: {analysis_report['subdomain_summary']['total_repositories_analyzed']}")
    logger.info(f"   - 引用关系总数: {analysis_report['subdomain_summary']['total_references_analyzed']}")
    logger.info(f"   - 引用类型总数: {len(all_ref_types)}")

    if chi2_results and 'chi2_statistic' in chi2_results:
        logger.info(f"2. 统计检验结果:")
        logger.info(f"   - 卡方统计量: χ² = {chi2_results['chi2_statistic']:.4f}")
        logger.info(f"   - 自由度: df = {chi2_results['degrees_of_freedom']}")
        logger.info(f"   - p值: p = {chi2_results['p_value']:.4e}")
        logger.info(f"   - 效应量 (Cramer's V): {chi2_results.get('effect_size_cramers_v', 0):.4f}")

        if chi2_results['significant']:
            logger.info(f"   - 结论: 不同子领域的引用类型分布存在显著差异 (p < 0.05)")
        else:
            logger.info(f"   - 结论: 不同子领域的引用类型分布无显著差异")

    logger.info(f"3. 各子领域主要引用类型:")
    for category_label, data in analysis_report['top_reference_types_by_subdomain'].items():
        logger.info(f"   - {category_label} ({data['repo_count']}个仓库, {data['total_references']}条引用):")
        for ref_type, count in list(data['top_types'].items())[:3]:
            proportion = count / data['total_references']
            logger.info(f"     * {ref_type}: {count} ({proportion:.1%})")

    logger.info("=" * 70)
    logger.info("子领域引用类型差异分析完成。")

    return {
        'analysis_report': analysis_report,
        'comparison_data': df_comparison,
        'subdomain_ref_data': subdomain_ref_data,
        'chi2_results': chi2_results
    }


def compare_subdomains_self_ref_time_evolution(split_dfs_by_category_label, df_self_ref_ratio, output_dir=None):
    """
    比较不同子领域（category_label）的外引比例随时间变化趋势

    参数:
    split_dfs_by_category_label: 按category_label划分的DataFrame字典 {label: df, ...}
    df_self_ref_ratio: 自引率数据DataFrame
    output_dir: 输出目录路径

    返回:
    dict: 包含子领域时间演化分析结果的字典
    """
    import numpy as np
    from scipy import stats

    logger.info("开始分析不同子领域的外引比例时间演化趋势...")

    if output_dir is None:
        github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
        output_dir = os.path.join(github_osdb_data_dir, "analysis_results/subdomain_time_evolution")

    os.makedirs(output_dir, exist_ok=True)

    # 确保输入数据有效
    if not split_dfs_by_category_label or df_self_ref_ratio.empty or df_repo_i_pr_rec_cnt.empty:
        logger.warning("输入数据为空，无法进行分析")
        return {}

    # 1. 准备数据：将仓库映射到子领域
    # 获取repo_name到category_label的映射
    repo_to_category = {}
    for category_label, df_category in split_dfs_by_category_label.items():
        for _, row in df_category.iterrows():
            repo_name = row['repo_name']
            if pd.notna(repo_name):
                repo_to_category[repo_name] = category_label

    logger.info(f"共映射了 {len(repo_to_category)} 个仓库到子领域")

    # 2. 合并数据：自引率 + 仓库信息 + 子领域标签
    # 确保数据类型一致
    df_repo_i_pr_rec_cnt["repo_id"] = df_repo_i_pr_rec_cnt["repo_id"].astype(str)
    df_self_ref_ratio["repo_id"] = df_self_ref_ratio["repo_id"].astype(str)

    # 过滤掉repo_name和repo_created_at同时为空的记录
    mask_not_null = df_repo_i_pr_rec_cnt["repo_name"].notna() & df_repo_i_pr_rec_cnt["repo_created_at"].notna()
    df_repo_info_clean = df_repo_i_pr_rec_cnt[mask_not_null].copy()

    # 检查去除空值后的唯一性
    if df_repo_info_clean["repo_name"].duplicated().any():
        logger.warning("去重后repo_name仍存在重复值，保留第一条记录")
        df_repo_info_clean = df_repo_info_clean.drop_duplicates(subset=['repo_name'], keep='first')

    if df_repo_info_clean["repo_id"].duplicated().any():
        logger.warning("去重后repo_id仍存在重复值，保留第一条记录")
        df_repo_info_clean = df_repo_info_clean.drop_duplicates(subset=['repo_id'], keep='first')

    logger.info(f"有效项目数量（有创建时间）: {len(df_repo_info_clean)}")

    # 合并自引率和项目信息
    df_merged = pd.merge(df_self_ref_ratio, df_repo_info_clean, on=['repo_id', 'repo_name'], how='inner')

    if len(df_merged) == 0:
        logger.warning("没有找到匹配的项目数据来分析子领域时间演化")
        return {}

    logger.info(f"成功匹配 {len(df_merged)} 个项目的数据")

    # 添加子领域标签
    df_merged['category_label'] = df_merged['repo_name'].map(repo_to_category)
    df_merged = df_merged[df_merged['category_label'].notna()]

    if len(df_merged) == 0:
        logger.warning("没有找到有子领域标签的项目数据")
        return {}

    logger.info(f"有子领域标签的项目数量: {len(df_merged)}")

    # 3. 计算项目年龄和外引比例（参考analyze_self_ref_time_evolution的实现）
    def calculate_project_age(created_at_str, reference_year=2023):
        """计算项目年龄（到参考年份年底的年数）"""
        try:
            # 转换时间戳，统一时区处理
            created_at = pd.to_datetime(created_at_str)

            # 如果时间戳带有时区，转换为UTC并移除时区信息
            if created_at.tz is not None:
                created_at = created_at.tz_convert('UTC').tz_localize(None)

            # 创建参考时间（2023年底），确保不带时区
            end_of_year = pd.Timestamp(f'{reference_year}-12-31')

            # 计算天数差
            age_days = (end_of_year - created_at).days
            age_years = age_days / 365.25  # 转换为年
            return age_years
        except Exception as e:
            logger.error(f"计算项目年龄时出错: {e}, 原始数据: {created_at_str}")
            return None

    df_merged['project_age_years'] = df_merged['repo_created_at'].apply(
        lambda x: calculate_project_age(x, reference_year=2023)
    )

    # 移除无法计算年龄的记录
    df_merged = df_merged.dropna(subset=['project_age_years'])

    if len(df_merged) == 0:
        logger.warning("没有有效的项目年龄数据")
        return None

    # 计算外引比率
    df_merged['external_ref_ratio'] = 1 - df_merged['self_ref_ratio']

    logger.info(f"最终分析数据集大小: {len(df_merged)} 个项目")
    logger.info(f"子领域分布: {df_merged['category_label'].value_counts().to_dict()}")

    # 4. 按子领域分组分析
    subdomain_analysis = {}
    categories = df_merged['category_label'].unique()

    for category in categories:
        df_subdomain = df_merged[df_merged['category_label'] == category]

        if len(df_subdomain) >= 3:  # 至少需要3个数据点进行有意义分析
            # 计算基本统计
            age_stats = {
                'count': len(df_subdomain),
                'mean_age': float(df_subdomain['project_age_years'].mean()),
                'median_age': float(df_subdomain['project_age_years'].median()),
                'min_age': float(df_subdomain['project_age_years'].min()),
                'max_age': float(df_subdomain['project_age_years'].max())
            }

            external_ref_stats = {
                'mean_external_ref': float(df_subdomain['external_ref_ratio'].mean()),
                'median_external_ref': float(df_subdomain['external_ref_ratio'].median()),
                'std_external_ref': float(df_subdomain['external_ref_ratio'].std()),
                'min_external_ref': float(df_subdomain['external_ref_ratio'].min()),
                'max_external_ref': float(df_subdomain['external_ref_ratio'].max())
            }

            # 计算年龄与外引比例的相关性
            if len(df_subdomain) >= 5:
                try:
                    correlation, p_value = stats.pearsonr(
                        df_subdomain['project_age_years'],
                        df_subdomain['external_ref_ratio']
                    )

                    # 线性回归分析
                    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
                        df_subdomain['project_age_years'],
                        df_subdomain['external_ref_ratio']
                    )

                    correlation_stats = {
                        'pearson_correlation': float(correlation),
                        'p_value': float(p_value),
                        'regression_slope': float(slope),
                        'regression_intercept': float(intercept),
                        'r_squared': float(r_value ** 2),
                        'significant': bool(p_value < 0.05)
                    }
                except Exception as e:
                    logger.warning(f"子领域 {category} 相关性分析失败: {str(e)}")
                    correlation_stats = None
            else:
                correlation_stats = None

            subdomain_analysis[category] = {
                'age_statistics': age_stats,
                'external_ref_statistics': external_ref_stats,
                'correlation_analysis': correlation_stats,
                'sample_size': len(df_subdomain)
            }
        else:
            logger.warning(f"子领域 {category} 数据点不足 ({len(df_subdomain)}个)，跳过详细分析")
            subdomain_analysis[category] = {
                'sample_size': len(df_subdomain),
                'insufficient_data': True
            }

    # 5. 可视化分析
    plt.figure(figsize=(18, 12))

    # 子图1: 各子领域外引比例与项目年龄的散点图（综合对比）
    plt.subplot(2, 3, 1)

    if len(df_merged) > 0:
        # 为每个子领域分配颜色
        unique_categories = df_merged['category_label'].unique()
        colormap = plt.cm.tab20
        category_colors = {cat: colormap(i % 20) for i, cat in enumerate(unique_categories)}

        # 绘制散点图
        for category in unique_categories:
            df_subdomain = df_merged[df_merged['category_label'] == category]
            if len(df_subdomain) > 0:
                plt.scatter(df_subdomain['project_age_years'],
                            df_subdomain['external_ref_ratio'],
                            color=category_colors[category],
                            s=40, alpha=0.7, edgecolors='black', linewidth=0.5,
                            label=f'{category} (n={len(df_subdomain)})')

        plt.xlabel('Project Age (years)', fontsize=11)
        plt.ylabel('External Reference Ratio', fontsize=11)
        plt.title('External Reference Ratio vs Project Age\nby Subdomain (2023)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')

        # 添加图例（放在右上角）
        plt.legend(title='Subdomain', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        # 添加总体趋势线
        if len(df_merged) >= 5:
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_merged['project_age_years'],
                    df_merged['external_ref_ratio']
                )

                x_range = np.linspace(df_merged['project_age_years'].min(),
                                      df_merged['project_age_years'].max(), 100)
                y_pred = slope * x_range + intercept

                plt.plot(x_range, y_pred, 'k-', linewidth=2, linestyle='--',
                         label=f'Overall trend (r={r_value:.3f})')

                plt.legend(title='Subdomain', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            except:
                pass
    else:
        plt.text(0.5, 0.5, 'No data available for scatter plot',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('External Reference vs Project Age by Subdomain', fontsize=13)

    # 子图2: 各子领域平均外引比例比较（条形图）
    plt.subplot(2, 3, 2)

    if subdomain_analysis:
        categories_with_stats = []
        mean_external_refs = []
        sample_sizes = []

        for category, analysis in subdomain_analysis.items():
            if 'external_ref_statistics' in analysis:
                categories_with_stats.append(category)
                mean_external_refs.append(analysis['external_ref_statistics']['mean_external_ref'])
                sample_sizes.append(analysis['sample_size'])

        if categories_with_stats:
            x_pos = np.arange(len(categories_with_stats))
            bars = plt.bar(x_pos, mean_external_refs, alpha=0.7,
                           color=plt.cm.tab20(range(len(categories_with_stats))))

            plt.xlabel('Subdomain', fontsize=11)
            plt.ylabel('Mean External Reference Ratio', fontsize=11)
            plt.title('Mean External Reference Ratio by Subdomain', fontsize=13, fontweight='bold')
            plt.xticks(x_pos, categories_with_stats, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')

            # 添加数值标签和样本大小
            for i, (bar, mean_val, n) in enumerate(zip(bars, mean_external_refs, sample_sizes)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{mean_val:.3f}\n(n={n})',
                         ha='center', va='bottom', fontsize=8)

            # 添加水平参考线（总体平均值）
            if len(mean_external_refs) > 0:
                overall_mean = np.mean(mean_external_refs)
                plt.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7,
                            label=f'Overall mean: {overall_mean:.3f}')
                plt.legend(fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No subdomain statistics available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Mean External Reference by Subdomain', fontsize=13)

    # 子图3: 各子领域年龄分布箱线图
    plt.subplot(2, 3, 3)

    if len(df_merged) > 0:
        # 创建年龄分布箱线图
        age_data = []
        age_labels = []

        for category in unique_categories:
            df_subdomain = df_merged[df_merged['category_label'] == category]
            if len(df_subdomain) >= 3:
                age_data.append(df_subdomain['project_age_years'].values)
                age_labels.append(f'{category}\n(n={len(df_subdomain)})')

        if age_data:
            box = plt.boxplot(age_data, labels=age_labels, patch_artist=True)

            # 设置箱体颜色
            colors = plt.cm.tab20(range(len(age_data)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            plt.xlabel('Subdomain', fontsize=11)
            plt.ylabel('Project Age (years)', fontsize=11)
            plt.title('Project Age Distribution by Subdomain', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')

            # 添加平均值点
            for i, ages in enumerate(age_data, 1):
                mean_age = np.mean(ages)
                plt.plot(i, mean_age, 'ro', markersize=6, label='Mean' if i == 1 else "")

            if len(age_data) > 0:
                plt.legend(['Mean'], loc='upper right', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No age distribution data available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Project Age Distribution by Subdomain', fontsize=13)

    # 子图4: 各子领域外引比例分布箱线图
    plt.subplot(2, 3, 4)

    if len(df_merged) > 0:
        # 创建外引比例分布箱线图
        ext_ref_data = []
        ext_ref_labels = []

        for category in unique_categories:
            df_subdomain = df_merged[df_merged['category_label'] == category]
            if len(df_subdomain) >= 3:
                ext_ref_data.append(df_subdomain['external_ref_ratio'].values)
                ext_ref_labels.append(f'{category}\n(n={len(df_subdomain)})')

        if ext_ref_data:
            box = plt.boxplot(ext_ref_data, labels=ext_ref_labels, patch_artist=True)

            # 设置箱体颜色
            colors = plt.cm.tab20(range(len(ext_ref_data)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            plt.xlabel('Subdomain', fontsize=11)
            plt.ylabel('External Reference Ratio', fontsize=11)
            plt.title('External Reference Ratio Distribution\nby Subdomain', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')

            # 添加平均值点
            for i, ext_refs in enumerate(ext_ref_data, 1):
                mean_ext_ref = np.mean(ext_refs)
                plt.plot(i, mean_ext_ref, 'ko', markersize=6, label='Mean' if i == 1 else "")

            if len(ext_ref_data) > 0:
                plt.legend(['Mean'], loc='upper right', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No external reference data available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('External Reference Distribution by Subdomain', fontsize=13)

    # 子图5: 各子领域年龄-外引比例相关性热图
    plt.subplot(2, 3, 5)

    if subdomain_analysis:
        categories_corr = []
        correlations = []
        p_values = []
        sample_sizes = []

        for category, analysis in subdomain_analysis.items():
            if (analysis.get('correlation_analysis') and
                    'pearson_correlation' in analysis['correlation_analysis']):
                categories_corr.append(category)
                correlations.append(analysis['correlation_analysis']['pearson_correlation'])
                p_values.append(analysis['correlation_analysis']['p_value'])
                sample_sizes.append(analysis['sample_size'])

        if categories_corr:
            # 创建热图数据
            corr_matrix = np.array(correlations).reshape(-1, 1)  # 单列热图

            # 创建热图
            im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto',
                            vmin=-1, vmax=1)

            plt.xlabel('Correlation Strength', fontsize=11)
            plt.ylabel('Subdomain', fontsize=11)
            plt.title('Age-External Reference Correlation\nby Subdomain', fontsize=13, fontweight='bold')

            # 设置y轴刻度
            plt.yticks(range(len(categories_corr)), categories_corr)

            # 添加颜色条
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('Pearson Correlation (r)', fontsize=10)

            # 在每个单元格中添加具体数值
            for i, (corr, p_val, n) in enumerate(zip(correlations, p_values, sample_sizes)):
                color = 'white' if abs(corr) > 0.5 else 'black'
                significance = '*' if p_val < 0.05 else ''
                plt.text(0, i, f'{corr:.3f}{significance}\nn={n}',
                         ha='center', va='center', color=color, fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='gray', alpha=0.3))

            plt.xticks([])  # 隐藏x轴刻度
    else:
        plt.text(0.5, 0.5, 'No correlation analysis data available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Age-External Reference Correlation by Subdomain', fontsize=13)

    # 子图6: 时间演化趋势对比（各子领域的回归线）
    plt.subplot(2, 3, 6)

    if subdomain_analysis:
        plt.figure(figsize=(10, 8))

        # 为每个有足够数据的子领域绘制回归线
        for category, analysis in subdomain_analysis.items():
            if (analysis.get('correlation_analysis') and
                    'regression_slope' in analysis['correlation_analysis'] and
                    analysis['sample_size'] >= 5):

                # 获取该子领域的数据
                df_subdomain = df_merged[df_merged['category_label'] == category]

                if len(df_subdomain) >= 5:
                    # 绘制数据点
                    plt.scatter(df_subdomain['project_age_years'],
                                df_subdomain['external_ref_ratio'],
                                s=20, alpha=0.5,
                                label=f'{category} (n={len(df_subdomain)})')

                    # 绘制回归线
                    slope = analysis['correlation_analysis']['regression_slope']
                    intercept = analysis['correlation_analysis']['regression_intercept']
                    r_squared = analysis['correlation_analysis']['r_squared']

                    x_range = np.linspace(df_subdomain['project_age_years'].min(),
                                          df_subdomain['project_age_years'].max(), 100)
                    y_pred = slope * x_range + intercept

                    plt.plot(x_range, y_pred, linewidth=2, alpha=0.8,
                             label=f'{category}: r²={r_squared:.3f}')

        plt.xlabel('Project Age (years)', fontsize=11)
        plt.ylabel('External Reference Ratio', fontsize=11)
        plt.title('Time Evolution Trends by Subdomain\n(Regression Lines)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.text(0.5, 0.5, 'No trend analysis data available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Time Evolution Trends by Subdomain', fontsize=13)

    plt.suptitle(
        'Subdomain Time Evolution Analysis: External Reference Ratio Trends\n(DBMS Projects by Category Label, 2023)',
        fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存可视化结果
    visualization_path = os.path.join(output_dir, "subdomain_time_evolution_analysis.png")
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"子领域时间演化分析可视化已保存至: {visualization_path}")

    # 6. 保存详细数据
    # 保存合并后的数据
    merged_csv_path = os.path.join(output_dir, "subdomain_time_evolution_data.csv")
    df_merged.to_csv(merged_csv_path, index=False, encoding='utf-8')
    logger.info(f"子领域时间演化数据已保存至: {merged_csv_path}")

    # 保存子领域分析结果
    analysis_json_path = os.path.join(output_dir, "subdomain_time_evolution_analysis.json")
    with open(analysis_json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(subdomain_analysis), f, indent=4, ensure_ascii=False)
    logger.info(f"子领域时间演化分析结果已保存至: {analysis_json_path}")

    # 7. 生成分析报告
    analysis_report = {
        'analysis_summary': {
            'total_projects_analyzed': len(df_merged),
            'subdomains_analyzed': list(df_merged['category_label'].unique()),
            'subdomain_distribution': df_merged['category_label'].value_counts().to_dict(),
            'overall_statistics': {
                'mean_project_age': float(df_merged['project_age_years'].mean()),
                'median_project_age': float(df_merged['project_age_years'].median()),
                'mean_external_ref_ratio': float(df_merged['external_ref_ratio'].mean()),
                'median_external_ref_ratio': float(df_merged['external_ref_ratio'].median())
            }
        },
        'subdomain_detailed_analysis': subdomain_analysis,
        'trend_comparison': {}
    }

    # 比较各子领域的趋势
    trends_comparison = []
    for category, analysis in subdomain_analysis.items():
        if analysis.get('correlation_analysis'):
            trends_comparison.append({
                'subdomain': category,
                'sample_size': analysis['sample_size'],
                'correlation': analysis['correlation_analysis']['pearson_correlation'],
                'r_squared': analysis['correlation_analysis']['r_squared'],
                'slope': analysis['correlation_analysis']['regression_slope'],
                'significant': analysis['correlation_analysis']['significant']
            })

    if trends_comparison:
        df_trends = pd.DataFrame(trends_comparison)
        df_trends = df_trends.sort_values('correlation', ascending=False)
        analysis_report['trend_comparison'] = df_trends.to_dict('records')

        # 保存趋势对比数据
        trends_csv_path = os.path.join(output_dir, "subdomain_trends_comparison.csv")
        df_trends.to_csv(trends_csv_path, index=False, encoding='utf-8')
        logger.info(f"子领域趋势对比数据已保存至: {trends_csv_path}")

    # 保存分析报告
    report_path = os.path.join(output_dir, "subdomain_time_evolution_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(analysis_report), f, indent=4, ensure_ascii=False)

    # 8. 打印关键发现
    logger.info("=" * 70)
    logger.info("子领域时间演化分析关键发现:")
    logger.info(f"1. 分析概况:")
    logger.info(f"   - 分析项目总数: {len(df_merged)}")
    logger.info(f"   - 涉及子领域数: {len(df_merged['category_label'].unique())}")
    logger.info(f"   - 平均项目年龄: {df_merged['project_age_years'].mean():.2f} 年")
    logger.info(f"   - 平均外引比例: {df_merged['external_ref_ratio'].mean():.4f}")

    logger.info(f"2. 子领域统计:")
    for category, analysis in subdomain_analysis.items():
        if 'external_ref_statistics' in analysis:
            stats = analysis['external_ref_statistics']
            logger.info(f"   - {category} (n={analysis['sample_size']}):")
            logger.info(f"     * 平均外引比例: {stats['mean_external_ref']:.4f}")
            logger.info(f"     * 中位数: {stats['median_external_ref']:.4f}")
            logger.info(f"     * 标准差: {stats['std_external_ref']:.4f}")

    logger.info(f"3. 时间演化趋势:")
    for trend in analysis_report.get('trend_comparison', []):
        significance = "显著" if trend['significant'] else "不显著"
        direction = "正相关" if trend['slope'] > 0 else "负相关" if trend['slope'] < 0 else "无关系"
        logger.info(f"   - {trend['subdomain']}: r={trend['correlation']:.3f}, "
                    f"r²={trend['r_squared']:.3f}, 斜率={trend['slope']:.4f} "
                    f"({direction}, {significance})")

    logger.info("=" * 70)
    logger.info("子领域时间演化分析完成。")

    return {
        'analysis_report': analysis_report,
        'merged_data': df_merged,
        'subdomain_analysis': subdomain_analysis
    }


def compare_subdomains_network_features(split_dfs_by_category_label, df_dbms_repos_ref_node_agg_dict,
                                        use_repo_nodes_only=True,
                                        output_dir=None, use_largest_component=True, only_dbms_repo=False, year=2023):
    """
    比较不同子领域（category_label）的网络特征差异

    参数:
    split_dfs_by_category_label: 按category_label划分的DataFrame字典 {label: df, ...}
    df_dbms_repos_ref_node_agg_dict: 引用关系字典 {repo_key: df_ref, ...}
    use_repo_nodes_only: 是否只使用Repo节点
    output_dir: 输出目录路径
    use_largest_component: 是否使用最大连通子图计算网络特征
    only_dbms_repo: 是否只分析DBMS仓库
    year: 分析的年份

    返回:
    dict: 包含子领域网络特征分析结果的字典
    """
    import numpy as np

    logger.info("开始分析不同子领域的网络特征差异...")

    if output_dir is None:
        github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
        output_dir = os.path.join(github_osdb_data_dir,
                                  f"analysis_results/subdomain_network_features{'_only_dbms_repo' if only_dbms_repo else ''}")

    os.makedirs(output_dir, exist_ok=True)

    # 确保输入数据有效
    if not split_dfs_by_category_label or not df_dbms_repos_ref_node_agg_dict:
        logger.warning("输入数据为空，无法进行分析")
        return {}

    # 1. 准备数据：将仓库映射到子领域
    repo_to_category = {}
    repo_key_to_category = {}
    for category_label, df_category in split_dfs_by_category_label.items():
        for _, row in df_category.iterrows():
            repo_name = row['repo_name']
            if pd.notna(repo_name):
                repo_name_fileformat = get_repo_name_fileformat(repo_name)
                filename = get_repo_year_filename(repo_name_fileformat, year)
                repo_key = filename.rstrip('.csv')
                repo_to_category[repo_name] = category_label
                repo_key_to_category[repo_key] = category_label

    logger.info(f"共映射了 {len(repo_to_category)} 个仓库到子领域")

    # 2. 按子领域构建网络并计算特征
    subdomain_network_features = {}

    # 首先，为每个子领域构建聚合网络
    subdomain_networks = {}
    for category_label in split_dfs_by_category_label.keys():
        subdomain_networks[category_label] = nx.MultiDiGraph()
        logger.info(f"初始化子领域 '{category_label}' 的网络")

    # 为每个仓库添加节点和边到对应的子领域网络
    for repo_key, df_ref in df_dbms_repos_ref_node_agg_dict.items():
        if repo_key not in repo_key_to_category:
            continue

        category_label = repo_key_to_category[repo_key]
        base_graph = subdomain_networks[category_label]

        if df_ref is not None and len(df_ref) > 0:
            try:
                # 过滤掉源节点或目标节点为空的记录
                df_ref_filtered = df_ref.dropna(subset=['src_entity_id_agg', 'tar_entity_id_agg'], how='any')

                if len(df_ref_filtered) == 0:
                    continue

                # 构建协作网络
                G = build_collab_net(df_ref_filtered,
                                     src_tar_colnames=['src_entity_id_agg', 'tar_entity_id_agg'],
                                     base_graph=base_graph,
                                     default_node_types=['src_entity_type_agg', 'tar_entity_type_agg'],
                                     default_edge_type="event_type",
                                     init_record_as_edge_attrs=True,
                                     use_df_col_as_default_type=True,
                                     out_g_type='DG')
                subdomain_networks[category_label] = G

                logger.debug(f"为子领域 '{category_label}' 添加了仓库 '{repo_key}' 的网络: {len(df_ref_filtered)} 条边")
            except Exception as e:
                logger.warning(f"为仓库 '{repo_key}' 构建网络失败: {str(e)}")

    # 如果只使用Repo节点，移除非Repo节点
    if use_repo_nodes_only:
        for category_label, G in subdomain_networks.items():
            nodes_to_remove = []
            for n, data in G.nodes(data=True):
                node_type = data.get('node_type', '')
                if node_type != 'Repo':
                    nodes_to_remove.append(n)

            if nodes_to_remove:
                logger.info(f"从子领域 '{category_label}' 的网络中移除 {len(nodes_to_remove)} 个非Repo节点")
                G.remove_nodes_from(nodes_to_remove)

    logger.info(f"成功构建了 {len(subdomain_networks)} 个子领域的聚合网络")

    # 3. 计算每个子领域网络的网络特征
    for category_label, G in subdomain_networks.items():
        logger.info(f"计算子领域 '{category_label}' 的网络特征...")

        if G.number_of_nodes() == 0:
            logger.warning(f"子领域 '{category_label}' 的网络没有节点，跳过")
            continue

        # 计算基本网络特征
        features = {
            'category_label': category_label,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0
        }

        # 计算平均度
        if G.number_of_nodes() > 0:
            degrees = [deg for _, deg in G.degree()]
            features['avg_degree'] = float(np.mean(degrees))
            features['max_degree'] = float(np.max(degrees)) if len(degrees) > 0 else 0
            features['min_degree'] = float(np.min(degrees)) if len(degrees) > 0 else 0
        else:
            features['avg_degree'] = 0
            features['max_degree'] = 0
            features['min_degree'] = 0

        # 计算平均聚类系数（转换为无向图）
        try:
            if G.number_of_edges() > 0:
                G_undirected = G.to_undirected()
                if G_undirected.number_of_edges() > 0:
                    clustering_coeffs = nx.clustering(G_undirected)
                    if clustering_coeffs:
                        avg_clustering = np.mean(list(clustering_coeffs.values()))
                        features['avg_clustering'] = float(avg_clustering)
                    else:
                        features['avg_clustering'] = 0
                else:
                    features['avg_clustering'] = 0
            else:
                features['avg_clustering'] = 0
        except Exception as e:
            logger.warning(f"计算聚类系数失败 {category_label}: {str(e)}")
            features['avg_clustering'] = 0

        # 计算平均路径长度（仅对连通图）
        try:
            if use_largest_component and not nx.is_connected(G.to_undirected()):
                # 获取最大连通子图
                connected_components = list(nx.connected_components(G.to_undirected()))
                if connected_components:
                    largest_component = max(connected_components, key=len)
                    G_lcc = G.subgraph(largest_component).to_undirected()

                    if G_lcc.number_of_nodes() > 1:
                        avg_path_length = nx.average_shortest_path_length(G_lcc)
                        features['avg_path_length'] = float(avg_path_length)
                        features['lcc_size'] = G_lcc.number_of_nodes()
                        features['lcc_coverage'] = G_lcc.number_of_nodes() / G.number_of_nodes()
                        features['num_components'] = len(connected_components)
                    else:
                        features['avg_path_length'] = 0
                        features['lcc_size'] = 0
                        features['lcc_coverage'] = 0
                        features['num_components'] = len(connected_components)
                else:
                    features['avg_path_length'] = 0
                    features['lcc_size'] = 0
                    features['lcc_coverage'] = 0
                    features['num_components'] = 0
            elif G.number_of_nodes() > 1 and nx.is_connected(G.to_undirected()):
                avg_path_length = nx.average_shortest_path_length(G.to_undirected())
                features['avg_path_length'] = float(avg_path_length)
                features['lcc_size'] = G.number_of_nodes()
                features['lcc_coverage'] = 1.0
                features['num_components'] = 1
            else:
                features['avg_path_length'] = 0
                features['lcc_size'] = 0
                features['lcc_coverage'] = 0
                features['num_components'] = len(list(nx.connected_components(G.to_undirected())))
        except Exception as e:
            logger.warning(f"计算平均路径长度失败 {category_label}: {str(e)}")
            features['avg_path_length'] = 0
            features['lcc_size'] = 0
            features['lcc_coverage'] = 0
            features['num_components'] = 0

        # 计算社区内引用比例（使用Louvain算法）
        try:
            if G.number_of_edges() > 0:
                G_undirected = G.to_undirected()

                # 尝试导入社区检测库
                try:
                    import community as community_louvain
                    partition = community_louvain.best_partition(G_undirected)

                    # 计算社区内引用比例
                    intra_community_edges = 0
                    total_edges = G_undirected.number_of_edges()

                    for u, v in G_undirected.edges():
                        if partition.get(u) == partition.get(v):
                            intra_community_edges += 1

                    if total_edges > 0:
                        intra_community_ratio = intra_community_edges / total_edges
                        features['intra_community_ratio'] = float(intra_community_ratio)
                        features['num_communities'] = len(set(partition.values()))

                        # 计算模块度
                        modularity = community_louvain.modularity(partition, G_undirected)
                        features['modularity'] = float(modularity)
                    else:
                        features['intra_community_ratio'] = 0
                        features['num_communities'] = 0
                        features['modularity'] = 0
                except ImportError:
                    # 使用NetworkX内置算法
                    import networkx.algorithms.community as nx_community
                    communities_generator = nx_community.greedy_modularity_communities(G_undirected)
                    communities = list(communities_generator)

                    # 转换为节点到社区ID的映射
                    partition = {}
                    for i, comm in enumerate(communities):
                        for node in comm:
                            partition[node] = i

                    # 计算社区内引用比例
                    intra_community_edges = 0
                    total_edges = G_undirected.number_of_edges()

                    for u, v in G_undirected.edges():
                        if partition.get(u) == partition.get(v):
                            intra_community_edges += 1

                    if total_edges > 0:
                        intra_community_ratio = intra_community_edges / total_edges
                        features['intra_community_ratio'] = float(intra_community_ratio)
                        features['num_communities'] = len(communities)

                        # 计算模块度
                        import networkx.algorithms.community.quality as nx_quality
                        modularity = nx_quality.modularity(G_undirected, communities)
                        features['modularity'] = float(modularity)
                    else:
                        features['intra_community_ratio'] = 0
                        features['num_communities'] = 0
                        features['modularity'] = 0
            else:
                features['intra_community_ratio'] = 0
                features['num_communities'] = 0
                features['modularity'] = 0
        except Exception as e:
            logger.warning(f"计算社区特征失败 {category_label}: {str(e)}")
            features['intra_community_ratio'] = 0
            features['num_communities'] = 0
            features['modularity'] = 0

        # 计算度同配性
        try:
            if G.number_of_edges() > 0:
                assortativity = nx.degree_assortativity_coefficient(G)
                features['assortativity'] = float(assortativity)
            else:
                features['assortativity'] = 0
        except Exception as e:
            logger.warning(f"计算度同配性失败 {category_label}: {str(e)}")
            features['assortativity'] = 0

        # 计算直径（最大连通子图）
        try:
            if use_largest_component and G.number_of_nodes() > 0:
                if not nx.is_connected(G.to_undirected()):
                    connected_components = list(nx.connected_components(G.to_undirected()))
                    if connected_components:
                        largest_component = max(connected_components, key=len)
                        G_lcc = G.subgraph(largest_component).to_undirected()
                        if G_lcc.number_of_nodes() > 1:
                            diameter = nx.diameter(G_lcc)
                            features['diameter'] = float(diameter)
                        else:
                            features['diameter'] = 0
                    else:
                        features['diameter'] = 0
                elif G.number_of_nodes() > 1:
                    diameter = nx.diameter(G.to_undirected())
                    features['diameter'] = float(diameter)
                else:
                    features['diameter'] = 0
            else:
                features['diameter'] = 0
        except Exception as e:
            logger.warning(f"计算直径失败 {category_label}: {str(e)}")
            features['diameter'] = 0

        subdomain_network_features[category_label] = features

    logger.info(f"成功计算了 {len(subdomain_network_features)} 个子领域的网络特征")

    # 4. 创建对比分析DataFrame
    if subdomain_network_features:
        df_comparison = pd.DataFrame(list(subdomain_network_features.values()))
        # 按节点数量排序
        df_comparison = df_comparison.sort_values('num_nodes', ascending=False).reset_index(drop=True)
    else:
        df_comparison = pd.DataFrame()
        logger.warning("没有有效的网络特征数据用于对比分析")

    # 5. 统计检验：检验不同子领域网络特征是否有显著差异
    anova_results = {}
    if len(subdomain_network_features) >= 2:
        try:
            # 准备特征数据
            feature_data = {}

            # 收集各子领域的特征数据
            for category_label, features in subdomain_network_features.items():
                # 直接使用特征值（每个子领域只有一个值）
                for feature_name in ['avg_degree', 'avg_clustering', 'avg_path_length',
                                     'intra_community_ratio', 'modularity', 'density', 'assortativity']:
                    if feature_name in features:
                        value = features[feature_name]
                        if feature_name not in feature_data:
                            feature_data[feature_name] = {}
                        if category_label not in feature_data[feature_name]:
                            feature_data[feature_name][category_label] = []
                        feature_data[feature_name][category_label].append(value)

            # 对每个特征进行ANOVA检验
            for feature_name, category_values in feature_data.items():
                if len(category_values) >= 2:
                    # 检查每个组是否有足够的数据
                    group_sizes = [len(values) for values in category_values.values()]
                    if min(group_sizes) >= 1 and len(category_values) >= 2:
                        try:
                            # 执行ANOVA检验
                            f_stat, p_value = stats.f_oneway(*list(category_values.values()))

                            anova_results[feature_name] = {
                                'f_statistic': float(f_stat),
                                'p_value': float(p_value),
                                'significant': bool(p_value < 0.05),
                                'num_groups': len(category_values),
                                'total_samples': sum(group_sizes)
                            }

                            logger.info(f"ANOVA检验 {feature_name}: F={f_stat:.4f}, p={p_value:.4e}")

                        except Exception as e:
                            logger.warning(f"ANOVA检验失败 {feature_name}: {str(e)}")
                            anova_results[feature_name] = {'error': str(e)}
        except Exception as e:
            logger.error(f"统计检验过程中出错: {str(e)}")
            traceback.print_exc()
    else:
        logger.warning("子领域数量不足，无法进行ANOVA检验")

    # 6. 可视化分析 (修改为英文标签)
    plt.figure(figsize=(20, 16))

    # 子图1: 各子领域网络规模对比（节点数和边数）
    ax1 = plt.subplot(3, 3, 1)

    if not df_comparison.empty:
        categories = df_comparison['category_label'].tolist()
        num_nodes = df_comparison['num_nodes'].tolist()
        num_edges = df_comparison['num_edges'].tolist()

        x_pos = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(x_pos - width / 2, num_nodes, width, alpha=0.7,
                        color='skyblue', label='Nodes')
        bars2 = ax1.bar(x_pos + width / 2, num_edges, width, alpha=0.7,
                        color='lightcoral', label='Edges')

        ax1.set_xlabel('Subdomain', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Network Scale: Nodes and Edges', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)

        # 缩短标签
        category_labels_short = []
        for cat in categories:
            if len(cat) > 10:
                category_labels_short.append(cat[:8] + '...')
            else:
                category_labels_short.append(cat)

        ax1.set_xticklabels(category_labels_short, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax1.text(bar1.get_x() + bar1.get_width() / 2., bar1.get_height(),
                     f'{int(bar1.get_height())}', ha='center', va='bottom', fontsize=8)
            ax1.text(bar2.get_x() + bar2.get_width() / 2., bar2.get_height(),
                     f'{int(bar2.get_height())}', ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No network scale data',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=11)
        ax1.set_title('Network Scale Comparison', fontsize=13)

    # 子图2: 各子领域平均度对比
    ax2 = plt.subplot(3, 3, 2)

    if not df_comparison.empty and 'avg_degree' in df_comparison.columns:
        categories = df_comparison['category_label'].tolist()
        avg_degrees = df_comparison['avg_degree'].tolist()

        x_pos = np.arange(len(categories))
        bars = ax2.bar(x_pos, avg_degrees, alpha=0.7,
                       color=plt.cm.Set3(range(len(categories))))

        ax2.set_xlabel('Subdomain', fontsize=11)
        ax2.set_ylabel('Average Degree', fontsize=11)
        ax2.set_title('Average Degree Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)

        category_labels_short = []
        for cat in categories:
            if len(cat) > 10:
                category_labels_short.append(cat[:8] + '...')
            else:
                category_labels_short.append(cat)

        ax2.set_xticklabels(category_labels_short, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, degree) in enumerate(zip(bars, avg_degrees)):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                     f'{degree:.2f}', ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No average degree data',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=11)
        ax2.set_title('Average Degree Comparison', fontsize=13)

    # 子图3: 各子领域平均聚类系数对比
    ax3 = plt.subplot(3, 3, 3)

    if not df_comparison.empty and 'avg_clustering' in df_comparison.columns:
        categories = df_comparison['category_label'].tolist()
        avg_clustering = df_comparison['avg_clustering'].tolist()

        x_pos = np.arange(len(categories))
        bars = ax3.bar(x_pos, avg_clustering, alpha=0.7,
                       color=plt.cm.Set3(range(len(categories))))

        ax3.set_xlabel('Subdomain', fontsize=11)
        ax3.set_ylabel('Avg Clustering Coefficient', fontsize=11)
        ax3.set_title('Clustering Coefficient Comparison', fontsize=13, fontweight='bold')
        ax3.set_xticks(x_pos)

        category_labels_short = []
        for cat in categories:
            if len(cat) > 10:
                category_labels_short.append(cat[:8] + '...')
            else:
                category_labels_short.append(cat)

        ax3.set_xticklabels(category_labels_short, rotation=45, ha='right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, clustering) in enumerate(zip(bars, avg_clustering)):
            ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{clustering:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No clustering coefficient data',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title('Clustering Coefficient Comparison', fontsize=13)

    # 子图4: 各子领域平均路径长度对比
    ax4 = plt.subplot(3, 3, 4)

    if not df_comparison.empty and 'avg_path_length' in df_comparison.columns:
        categories = df_comparison['category_label'].tolist()
        avg_path_lengths = df_comparison['avg_path_length'].tolist()

        # 过滤掉为0的值
        valid_indices = [i for i, val in enumerate(avg_path_lengths) if val > 0]
        if valid_indices:
            valid_categories = [categories[i] for i in valid_indices]
            valid_path_lengths = [avg_path_lengths[i] for i in valid_indices]

            x_pos = np.arange(len(valid_categories))
            bars = ax4.bar(x_pos, valid_path_lengths, alpha=0.7,
                           color=plt.cm.Set3(range(len(valid_categories))))

            ax4.set_xlabel('Subdomain', fontsize=11)
            ax4.set_ylabel('Average Path Length', fontsize=11)
            ax4.set_title('Average Path Length Comparison', fontsize=13, fontweight='bold')
            ax4.set_xticks(x_pos)

            category_labels_short = []
            for cat in valid_categories:
                if len(cat) > 10:
                    category_labels_short.append(cat[:8] + '...')
                else:
                    category_labels_short.append(cat)

            ax4.set_xticklabels(category_labels_short, rotation=45, ha='right', fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for i, (bar, path_len) in enumerate(zip(bars, valid_path_lengths)):
                ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                         f'{path_len:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No valid path length data',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=11)
            ax4.set_title('Average Path Length Comparison', fontsize=13)
    else:
        ax4.text(0.5, 0.5, 'No path length data',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.set_title('Average Path Length Comparison', fontsize=13)

    # 子图5: 各子领域社区内引用比例对比
    ax5 = plt.subplot(3, 3, 5)

    if not df_comparison.empty and 'intra_community_ratio' in df_comparison.columns:
        categories = df_comparison['category_label'].tolist()
        intra_ratios = df_comparison['intra_community_ratio'].tolist()

        x_pos = np.arange(len(categories))
        bars = ax5.bar(x_pos, intra_ratios, alpha=0.7,
                       color=plt.cm.Set3(range(len(categories))))

        ax5.set_xlabel('Subdomain', fontsize=11)
        ax5.set_ylabel('Intra-community Ratio', fontsize=11)
        ax5.set_title('Intra-community Reference Ratio', fontsize=13, fontweight='bold')
        ax5.set_xticks(x_pos)

        category_labels_short = []
        for cat in categories:
            if len(cat) > 10:
                category_labels_short.append(cat[:8] + '...')
            else:
                category_labels_short.append(cat)

        ax5.set_xticklabels(category_labels_short, rotation=45, ha='right', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, ratio) in enumerate(zip(bars, intra_ratios)):
            ax5.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'No intra-community ratio data',
                 ha='center', va='center', transform=ax5.transAxes, fontsize=11)
        ax5.set_title('Intra-community Reference Ratio', fontsize=13)

    # 子图6: 各子领域模块度对比
    ax6 = plt.subplot(3, 3, 6)

    if not df_comparison.empty and 'modularity' in df_comparison.columns:
        categories = df_comparison['category_label'].tolist()
        modularities = df_comparison['modularity'].tolist()

        x_pos = np.arange(len(categories))
        bars = ax6.bar(x_pos, modularities, alpha=0.7,
                       color=plt.cm.Set3(range(len(categories))))

        ax6.set_xlabel('Subdomain', fontsize=11)
        ax6.set_ylabel('Modularity', fontsize=11)
        ax6.set_title('Modularity Comparison', fontsize=13, fontweight='bold')
        ax6.set_xticks(x_pos)

        category_labels_short = []
        for cat in categories:
            if len(cat) > 10:
                category_labels_short.append(cat[:8] + '...')
            else:
                category_labels_short.append(cat)

        ax6.set_xticklabels(category_labels_short, rotation=45, ha='right', fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, modularity) in enumerate(zip(bars, modularities)):
            ax6.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{modularity:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'No modularity data',
                 ha='center', va='center', transform=ax6.transAxes, fontsize=11)
        ax6.set_title('Modularity Comparison', fontsize=13)

    # 子图7: 网络特征相关性热图
    ax7 = plt.subplot(3, 3, 7)

    if not df_comparison.empty and len(df_comparison) >= 3:
        # 选择网络特征列
        feature_cols = ['avg_degree', 'avg_clustering', 'avg_path_length',
                        'intra_community_ratio', 'modularity', 'density', 'assortativity']
        available_cols = [col for col in feature_cols if col in df_comparison.columns]

        if len(available_cols) >= 2:
            # 创建特征矩阵
            feature_matrix = df_comparison[available_cols].values

            # 计算特征相关性
            correlation_matrix = np.corrcoef(feature_matrix, rowvar=False)

            # 绘制热图
            im = ax7.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

            # 设置坐标轴标签
            feature_names_short = []
            for col in available_cols:
                if len(col) > 10:
                    feature_names_short.append(col[:8] + '...')
                else:
                    feature_names_short.append(col)

            ax7.set_xticks(range(len(feature_names_short)))
            ax7.set_yticks(range(len(feature_names_short)))
            ax7.set_xticklabels(feature_names_short, rotation=45, ha='right', fontsize=8)
            ax7.set_yticklabels(feature_names_short, fontsize=8)
            ax7.set_title('Network Feature Correlation', fontsize=13, fontweight='bold')

            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax7, shrink=0.8)
            cbar.set_label('Correlation Coefficient', fontsize=10)

            # 添加相关系数值
            for i in range(len(feature_names_short)):
                for j in range(len(feature_names_short)):
                    corr_value = correlation_matrix[i, j]
                    color = 'white' if abs(corr_value) > 0.5 else 'black'
                    ax7.text(j, i, f'{corr_value:.2f}', ha='center', va='center',
                             color=color, fontsize=7)
        else:
            ax7.text(0.5, 0.5, 'Insufficient features\nfor correlation analysis',
                     ha='center', va='center', transform=ax7.transAxes, fontsize=11)
            ax7.set_title('Network Feature Correlation', fontsize=13)
    else:
        ax7.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis',
                 ha='center', va='center', transform=ax7.transAxes, fontsize=11)
        ax7.set_title('Network Feature Correlation', fontsize=13)

    # 子图8: 网络密度对比
    ax8 = plt.subplot(3, 3, 8)

    if not df_comparison.empty and 'density' in df_comparison.columns:
        categories = df_comparison['category_label'].tolist()
        densities = df_comparison['density'].tolist()

        x_pos = np.arange(len(categories))
        bars = ax8.bar(x_pos, densities, alpha=0.7,
                       color=plt.cm.Set3(range(len(categories))))

        ax8.set_xlabel('Subdomain', fontsize=11)
        ax8.set_ylabel('Network Density', fontsize=11)
        ax8.set_title('Network Density Comparison', fontsize=13, fontweight='bold')
        ax8.set_xticks(x_pos)

        category_labels_short = []
        for cat in categories:
            if len(cat) > 10:
                category_labels_short.append(cat[:8] + '...')
            else:
                category_labels_short.append(cat)

        ax8.set_xticklabels(category_labels_short, rotation=45, ha='right', fontsize=9)
        ax8.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, density) in enumerate(zip(bars, densities)):
            ax8.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0001,
                     f'{density:.4f}', ha='center', va='bottom', fontsize=8)
    else:
        ax8.text(0.5, 0.5, 'No network density data',
                 ha='center', va='center', transform=ax8.transAxes, fontsize=11)
        ax8.set_title('Network Density Comparison', fontsize=13)

    # 子图9: 子领域网络可视化示例（最大子领域）
    ax9 = plt.subplot(3, 3, 9)

    if not df_comparison.empty:
        # 找到节点数最多的子领域
        max_nodes_idx = df_comparison['num_nodes'].idxmax()
        max_category = df_comparison.loc[max_nodes_idx, 'category_label']

        if max_category in subdomain_networks:
            G_max = subdomain_networks[max_category]

            if G_max.number_of_nodes() > 0:
                # 简化可视化：只显示节点和基本结构
                try:
                    # 如果节点太多，只显示节点分布
                    if G_max.number_of_nodes() > 50:
                        # 计算节点度分布
                        degrees = [deg for _, deg in G_max.degree()]
                        hist, bins = np.histogram(degrees, bins=20)

                        ax9.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.7, color='steelblue')
                        ax9.set_xlabel('Node Degree', fontsize=11)
                        ax9.set_ylabel('Frequency', fontsize=11)
                        ax9.set_title(f'Largest Network Degree Distribution\n({max_category})', fontsize=12,
                                      fontweight='bold')
                        ax9.grid(True, alpha=0.3, axis='y')
                    else:
                        # 绘制网络图
                        G_undirected = G_max.to_undirected()

                        # 使用spring layout布局
                        pos = nx.spring_layout(G_undirected, seed=42, k=1 / np.sqrt(G_undirected.number_of_nodes()))

                        # 绘制节点
                        nx.draw_networkx_nodes(G_undirected, pos, ax=ax9,
                                               node_size=50, node_color='lightblue',
                                               alpha=0.8)

                        # 绘制边
                        nx.draw_networkx_edges(G_undirected, pos, ax=ax9,
                                               alpha=0.5, edge_color='gray',
                                               width=0.5)

                        # 绘制标签（只显示部分节点）
                        node_labels = {}
                        for i, node in enumerate(G_undirected.nodes()):
                            if i < 10:  # 只显示前10个节点的标签
                                node_labels[node] = str(node)[:10]

                        nx.draw_networkx_labels(G_undirected, pos, node_labels, ax=ax9,
                                                font_size=7, font_color='darkred')

                        ax9.set_title(
                            f'Network Structure Visualization\n({max_category}, {G_max.number_of_nodes()} nodes)',
                            fontsize=12, fontweight='bold')
                        ax9.axis('off')

                        # 添加网络基本信息
                        info_text = f"Nodes: {G_max.number_of_nodes()}\nEdges: {G_max.number_of_edges()}"
                        ax9.text(0.02, 0.02, info_text, transform=ax9.transAxes,
                                 fontsize=9, verticalalignment='bottom',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                except Exception as e:
                    ax9.text(0.5, 0.5, f'Network visualization failed:\n{str(e)[:30]}...',
                             ha='center', va='center', transform=ax9.transAxes, fontsize=10)
                    ax9.set_title('Network Structure Visualization', fontsize=12)
            else:
                ax9.text(0.5, 0.5, f'Subdomain {max_category} network is empty',
                         ha='center', va='center', transform=ax9.transAxes, fontsize=11)
                ax9.set_title('Network Structure Visualization', fontsize=12)
        else:
            ax9.text(0.5, 0.5, 'Cannot find largest network',
                     ha='center', va='center', transform=ax9.transAxes, fontsize=11)
            ax9.set_title('Network Structure Visualization', fontsize=12)
    else:
        ax9.text(0.5, 0.5, 'No network data available',
                 ha='center', va='center', transform=ax9.transAxes, fontsize=11)
        ax9.set_title('Network Structure Visualization', fontsize=12)

    plt.tight_layout()

    # 保存可视化结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(output_dir, f'subdomain_network_features_comparison_{timestamp}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved network features comparison figure: {fig_path}")

    # 7. 保存详细的网络特征数据
    if df_comparison is not None and not df_comparison.empty:
        # 保存DataFrame为CSV
        csv_path = os.path.join(output_dir, f'subdomain_network_features_{timestamp}.csv')
        df_comparison.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"已保存网络特征数据: {csv_path}")

        # 保存JSON格式的详细结果
        json_results = {
            'timestamp': timestamp,
            'year': year,
            'use_repo_nodes_only': use_repo_nodes_only,
            'use_largest_component': use_largest_component,
            'only_dbms_repo': only_dbms_repo,
            'subdomain_count': len(subdomain_network_features),
            'subdomain_features': subdomain_network_features,
            'comparison_table': df_comparison.to_dict('records'),
            'anova_results': anova_results,
            'repo_to_category_mapping': {k: v for i, (k, v) in enumerate(repo_to_category.items()) if i < 100}
            # 只保存前100个映射
        }

        # 转换NumPy类型
        json_results = convert_numpy_types(json_results)

        json_path = os.path.join(output_dir, f'subdomain_network_features_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存详细结果: {json_path}")

    # 8. 生成统计摘要 (英文)
    logger.info("=" * 80)
    logger.info("Subdomain Network Features Analysis Summary:")
    logger.info(f"Analysis Year: {year}")
    logger.info(f"Number of Subdomains: {len(subdomain_network_features)}")
    logger.info(f"Use Repo Nodes Only: {use_repo_nodes_only}")
    logger.info(f"Analyze DBMS Repos Only: {only_dbms_repo}")

    if df_comparison is not None and not df_comparison.empty:
        # 找到节点数最多的子领域
        max_nodes_row = df_comparison.loc[df_comparison['num_nodes'].idxmax()]
        min_nodes_row = df_comparison.loc[df_comparison['num_nodes'].idxmin()]

        logger.info("Network Scale Statistics:")
        logger.info(
            f"  Total Nodes Range: {min_nodes_row['num_nodes']} - {max_nodes_row['num_nodes']} (Avg: {df_comparison['num_nodes'].mean():.1f})")
        logger.info(
            f"  Total Edges Range: {min_nodes_row['num_edges']} - {df_comparison['num_edges'].max()} (Avg: {df_comparison['num_edges'].mean():.1f})")

        # 网络特征统计
        if 'avg_degree' in df_comparison.columns:
            logger.info(
                f"  Average Degree Range: {df_comparison['avg_degree'].min():.2f} - {df_comparison['avg_degree'].max():.2f} (Avg: {df_comparison['avg_degree'].mean():.2f})")

        if 'avg_clustering' in df_comparison.columns:
            logger.info(
                f"  Avg Clustering Coefficient Range: {df_comparison['avg_clustering'].min():.3f} - {df_comparison['avg_clustering'].max():.3f} (Avg: {df_comparison['avg_clustering'].mean():.3f})")

        if 'modularity' in df_comparison.columns:
            logger.info(
                f"  Modularity Range: {df_comparison['modularity'].min():.3f} - {df_comparison['modularity'].max():.3f} (Avg: {df_comparison['modularity'].mean():.3f})")

        # ANOVA检验结果摘要
        if anova_results:
            significant_features = [feat for feat, result in anova_results.items()
                                    if isinstance(result, dict) and result.get('significant', False)]

            if significant_features:
                logger.info(
                    f"  ANOVA found {len(significant_features)} features with significant differences: {', '.join(significant_features)}")
            else:
                logger.info("  ANOVA found no significant differences between groups")

        # 输出最大和最小网络的详细信息
        logger.info("\nLargest Network (by node count):")
        logger.info(f"  Subdomain: {max_nodes_row['category_label']}")
        logger.info(f"  Nodes: {int(max_nodes_row['num_nodes'])}")
        logger.info(f"  Edges: {int(max_nodes_row['num_edges'])}")
        logger.info(f"  Density: {max_nodes_row.get('density', 'N/A'):.6f}")

        logger.info("\nSmallest Network (by node count):")
        logger.info(f"  Subdomain: {min_nodes_row['category_label']}")
        logger.info(f"  Nodes: {int(min_nodes_row['num_nodes'])}")
        logger.info(f"  Edges: {int(min_nodes_row['num_edges'])}")
        logger.info(f"  Density: {min_nodes_row.get('density', 'N/A'):.6f}")

    logger.info("=" * 80)

    # 9. 返回分析结果
    results = {
        'subdomain_networks': subdomain_networks,
        'subdomain_features': subdomain_network_features,
        'comparison_dataframe': df_comparison,
        'anova_results': anova_results,
        'repo_to_category': repo_to_category,
        'repo_key_to_category': repo_key_to_category,
        'output_dir': output_dir,
        'visualization_path': fig_path
    }

    logger.info("Subdomain network features analysis completed")

    return results


def convert_numpy_types(obj):
    """将NumPy类型转换为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


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
    logger.info(f"-------------Step 2. Extract Relationship-------------")
    keep_part = 'is_not_file' if flag_skip_existing_files else 'all'
    filenames_need_extract = filenames_exist_filter(dbms_repos_dedup_content_dir, filenames, keep_part=keep_part)
    # Get repo_keys
    logger.info(f"Read data from {dbms_repos_dedup_content_dir}. This may take a lot of time...")
    df_dbms_repos_dict = read_csvs(dbms_repos_dedup_content_dir, filenames=filenames_need_extract, index_col=0)
    logger.info(f"Read completed.")

    d_repo_record_length = {k: len(df) for k, df in df_dbms_repos_dict.items()}
    d_repo_record_length_sorted = dict(sorted(d_repo_record_length.items(), key=lambda x: x[1], reverse=True))
    repo_keys_need_extract = list(d_repo_record_length_sorted.keys())
    df_dbms_repos_need_extract_dict = {k: df_dbms_repos_dict[k] for k in repo_keys_need_extract}
    logger.info(f"The {len(d_repo_record_length_sorted)} repo_keys to be processed sorted by the records count: {d_repo_record_length_sorted}")

    # Collaboration relation extraction
    logger.info(f"Collaboration relation extraction start...")
    collaboration_relation_extraction(repo_keys_need_extract, df_dbms_repos_need_extract_dict, dbms_repos_gh_core_dir,
                                      update_exists=not flag_skip_existing_files, add_mode_if_exists=True,
                                      use_relation_type_list=["EventAction", "Reference"], last_stop_index=-1)
    logger.info(f"Collaboration relation extraction completed.")

    # 步骤3: 引用耦合网络构建
    logger.info(f"-------------Step 3. Build Reference Network-------------")
    # 边ref去重、结点repo actor粒度聚合
    if not flag_skip_existing_files:
        # read relations
        df_dbms_repos_ref_dict = read_csvs(dbms_repos_gh_core_dir, filenames=filenames, index_col=None)
        # reference filter
        df_dbms_repos_ref_dict = {k: df[df["relation_type"] == "Reference"] for k, df in df_dbms_repos_ref_dict.items()}
        # deduplicate reference relation if ref_dedup_by_event_id = True: deduplicate the same notna <src_entity_id, tar_entity_id> pairs with a same event_id
        if ref_dedup_by_event_id:
            df_dbms_repos_ref_dict = {k: dedup_x_y_keep_na_by_z(df, subset=['src_entity_id', 'tar_entity_id', 'event_id'], keep='first') for k, df in df_dbms_repos_ref_dict.items()}
        # granularity aggregation
        df_dbms_repos_ref_node_agg_dict = {}
        for repo_key, df_dbms_repo_ref in list(df_dbms_repos_ref_dict.items()):
            # get repo_id by repo_key using df_target_repos
            repo_id = get_repo_id_by_repo_key(repo_key, df_target_repos, year)
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

    # add node attributes: "degree", "repo_name"
    degrees = dict(G_repo.degree())
    nx.set_node_attributes(G_repo, degrees, 'degree')

    df_target_repos["repo_id"] = df_target_repos["repo_id"].astype(str)
    df_filtered = df_target_repos.dropna(subset=['repo_id'])  # 去除key列为空值的行
    df_filtered = df_filtered.drop_duplicates(subset=['repo_id'], keep='first')  # 去除key列重复值的行，保留第一次出现的
    # show the repo_name in df_target_repos as node labels
    repo_id_name_dict = df_filtered.set_index('repo_id')['repo_name'].to_dict()
    for node in G_repo.nodes():
        node_str = str(node)
        if node_str.startswith("R_"):
            repo_id = node_str.lstrip("R_")
            G_repo.nodes[node]["repo_id"] = repo_id
            G_repo.nodes[node]["repo_name"] = repo_id_name_dict.get(repo_id, "")

    # filter nodes and edges
    only_dbms_repo = False
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

    # G_repo_ud = DG2G(G_repo, only_upper_triangle=False, multiplicity=True, double_self_loop=True)
    # feat = ["len_nodes", "len_edges", "edge_density", "is_sparse", "avg_deg", "avg_clustering",
    #         "lcc_node_coverage_ratio", "lcc_len_nodes", "lcc_len_edges", "lcc_edge_density", "lcc_diameter",
    #         "lcc_assort_coe", "lcc_avg_dist"]
    #
    # graph_feature_record_complex_network = get_graph_feature(G_repo_ud, feat=feat)
    # df_dbms_repos_ref_net_node_agg_feat = pd.DataFrame.from_dict(graph_feature_record_complex_network, orient='index')
    # feat_filename = f"homo{'_only' if only_dbms_repo else ''}_dbms_repos_ref_net_node_agg{'_dsl' if drop_self_loop else ''}_feat.csv"
    # df_dbms_repos_ref_net_node_agg_feat_path = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], f'analysis_results/{feat_filename}')
    # df_dbms_repos_ref_net_node_agg_feat.to_csv(df_dbms_repos_ref_net_node_agg_feat_path, header=False, index=True)
    # logger.info(f"{df_dbms_repos_ref_net_node_agg_feat_path} saved!")

    # # 步骤4: 描述性指标分析
    # logger.info(f"-------------Step 4. Descriptive indicator analysis-------------")
    # logger.info("""
    # 	 4.1 DBMS项目引用耦合模式的描述性分析（占全网络）
	# 	 4.1.1 引用类型分布特征
	# 		294个DBMS项目，施引涉及github通用服务unique项目数和用户数？
	# 		引用关系识别数：？条，
	# 		核心实体引用类型分布？
	# 			Actor
	# 			Issue/PR
	# 			Commit
	# 			File/Doc
	# 			...
	# 			GitHub_General_Service_Other_Links
	# 			GitHub_Other_Service
	# 			GitHub_Service_External_Links
	# 		自引率分布统计描述
	# 			平均数、中位数、最大、最小
	# 	 4.1.2 活跃议题数、新增议评率、新增评引率分布统计描述
	# 		活跃议题数（活跃Issue和pr数）
	# 		新增议评率（含Issue和PR body的新增Comment事件数/活跃Issue和pr数）
	# 		新增评引率（Comment事件中新增引用数/含Issue和PR body的新增Comment事件数）
	# 	 4.1.3 自引率时间演化特征
	# 		外引比率随项目年龄的散点图
    # """)
    logger.info(f"Read data from {dbms_repos_gh_core_ref_node_agg_dir} and {dbms_repos_gh_core_dir}. This may take a lot of time...")
    df_dbms_repos_ref_node_agg_dict = read_csvs(dbms_repos_gh_core_ref_node_agg_dir, filenames=filenames, index_col=0)
    # df_dbms_repos_ref_dict = read_csvs(dbms_repos_gh_core_dir, filenames=filenames, index_col=None)
    logger.info(f"Read completed.")
    github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
    repo_i_pr_rec_cnt_path = os.path.join(github_osdb_data_dir, 'repo_activity_statistics/repo_i_pr_rec_cnt.csv')
    df_repo_i_pr_rec_cnt = pd.read_csv(repo_i_pr_rec_cnt_path, index_col=None)
    # analyze_referenced_type_distribution(df_dbms_repos_ref_node_agg_dict, df_repo_i_pr_rec_cnt)
    # calculate_issue_referencing_metrics(df_dbms_repos_ref_dict, df_repo_i_pr_rec_cnt)
    df_self_ref_ratio = pd.read_csv(os.path.join(github_osdb_data_dir, f"analysis_results/ref_type_dist/df_self_ref_ratio.csv"), index_col=None)
    # analyze_self_ref_time_evolution(df_self_ref_ratio, df_repo_i_pr_rec_cnt)

    # # 步骤5: 网络拓扑特征分析
    # logger.info(f"-------------Step 5. Network topology characteristics Analysis-------------")
    # logger.info("""
    # 	 4.2 引用耦合网络的拓扑特征分析（github通用服务: row["tar_entity_type_fine_grained"] not in ["GitHub_Other_Service", "GitHub_Service_External_Links"]）
	# 	 4.2.1 度分布与无标度特性
	# 		DBMS项目引用耦合网络的度分布
	# 	 4.2.2 中心性与关键节点识别
	# 	 4.2.3 聚类系数与社区结构
	# 		Louvain算法对网络进行社区检测
    # """)
    rm_edges = [(u, v) for u, v, d in G_repo.edges(data=True) if d["tar_entity_type_fine_grained"] in ["GitHub_Other_Service", "GitHub_Service_External_Links"]]
    G_repo.remove_edges_from(rm_edges)
    # G_repo_ud = DG2G(G_repo, only_upper_triangle=False, multiplicity=True, double_self_loop=True)
    # analyze_degree_distribution(G_repo_ud, only_dbms_repo=only_dbms_repo)
    # calculate_centrality_measures(G_repo_ud, only_dbms_repo=only_dbms_repo)
    # detect_community_structure(G_repo_ud, only_dbms_repo=only_dbms_repo)
    # compute_clustering_coefficient(G_repo_ud, only_dbms_repo=only_dbms_repo)

    # 步骤6: 描述性指标分析
    logger.info(f"-------------Step 6. Analysis of differences in reference patterns among sub domains-------------")
    logger.info("""
    	 4.3 不同类型DBMS项目的引用模式差异分析
		 4.3.1 引用类型的领域差异
			子领域引用类型分布显著差异（χ²=1256.7, df=12, p<0.001）
		 4.3.2 时间演化的领域差异
			外引比例随时间变化趋势
		 4.3.3 网络参数的领域差异（github通用服务: row["tar_entity_type_fine_grained"] not in ["GitHub_Other_Service", "GitHub_Service_External_Links"]）
			网络参数：平均度 | 平均聚类系数 | 平均路径长度 | 社区内引用比例
    """)
    category_label_colname = "category_label"
    df_repo_i_pr_rec_cnt_filtered = df_repo_i_pr_rec_cnt[df_repo_i_pr_rec_cnt["repo_name"].isin(repo_names)]
    df_repo_i_pr_rec_cnt_filtered = df_repo_i_pr_rec_cnt_filtered[df_repo_i_pr_rec_cnt_filtered[category_label_colname].notna()]
    split_dfs_by_category_label = split_dataframe_by_column(df_repo_i_pr_rec_cnt_filtered, category_label_colname)
    # compare_subdomains_referenced_type(split_dfs_by_category_label, df_dbms_repos_ref_node_agg_dict)
    #  # compare_subdomains_self_ref_time_evolution(split_dfs_by_category_label, df_self_ref_ratio)  # 疑似独立
    compare_subdomains_network_features(split_dfs_by_category_label, df_dbms_repos_ref_node_agg_dict, only_dbms_repo=only_dbms_repo)

