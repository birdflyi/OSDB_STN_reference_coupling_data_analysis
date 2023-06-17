#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/4/19 21:14
# @Author : 'Lou Zehua'
# @File   : query_OSDB_github_log.py

import os
import sys

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

import numpy as np
import pandas as pd

from etc import filePathConf
from script import columns_simple, body_columns_dict, event_columns_dict, re_ref_patterns
from script.body_content_preprocessing import read_csvs, dedup_content
from script.identify_reference import drop_allNA, find_substrs_in_df_repos_ref_type_local_msg, dump_to_pickle, \
    load_pickle, substrs2rawstr_in_df_repos_ref_type_local_msg, substrs2rawstr_in_df_repos_all_ref_type_local_msg
from script.query_OSDB_github_log import query_repo_log_each_year_to_csv_dir
from script.statistic_analysis_reference_entities import stat_ref_wight_df_repos_ref_type_msg, \
    union_all_ref_type_df_substrscnt_for_repos, agg_df_dict, concat_df_dict


def query_OSDB_github_log_from_dbserver(update_exist_data=False):
    # 1. 按repo_name分散存储到每一个csv文件中
    UPDATE_EXIST_DATA = update_exist_data  # UPDATE SAVED RESULTS FLAG
    # 1.1 repo reference features as columns of sql
    columns = columns_simple
    # 1.2 get repo_names as condition of sql
    # repo_names = ['sqlite/sqlite', 'MariaDB/server', 'mongodb/mongo', 'redis/redis', 'elastic/elasticsearch', 'influxdata/influxdb', 'ClickHouse/ClickHouse', 'apache/hbase']
    df_OSDB_githubprj = pd.read_csv(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                                 "dbfeatfusion_records_202306_automerged_manulabeled_with_repoid.csv"),
                                    header='infer', index_col=None)
    df_OSDB_githubprj = df_OSDB_githubprj[
        pd.notna(df_OSDB_githubprj["github_repo_id"])]  # filter github_repo_id must exist
    repo_names = list(df_OSDB_githubprj["github_repo_link"].values)
    # 1.3 query and save
    save_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], "repos")
    sql_param = {
        "table": "opensource.gh_events",
        "start_end_year": [2022, 2023],
    }
    query_repo_log_each_year_to_csv_dir(repo_names, columns, save_dir, sql_param, update_exist_data=UPDATE_EXIST_DATA)
    return


def process_body_content(dedup_content_overwrite=False):
    # reduce_redundancy
    # 读入csv，去除数据库存储时额外复制的重复issue信息
    dbms_repos_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], 'repos')
    df_dbms_repos_raw_dict = read_csvs(dbms_repos_dir)
    print(len(df_dbms_repos_raw_dict))
    DEDUP_CONTENT_OVERWRITE = dedup_content_overwrite  # UPDATE SAVED RESULTS FLAG
    dbms_repos_dedup_content_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], 'repos_dedup_content')
    for repo_key, df_dbms_repo in df_dbms_repos_raw_dict.items():
        save_path = os.path.join(dbms_repos_dedup_content_dir, "{repo_key}.csv".format(**{"repo_key": repo_key}))
        if DEDUP_CONTENT_OVERWRITE or not os.path.exists(save_path):
            dedup_content(df_dbms_repo).to_csv(save_path)
    if not DEDUP_CONTENT_OVERWRITE:
        print('skip exist dedup_content...')
    print('dedup_content done!')
    return


def identify_reference_substrs(update_ref_msg_regexed_dict_pkl=False):
    # 读入csv，筛选项目
    dbms_repos_dedup_content_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                                'repos_dedup_content')
    df_dbms_repos_dict = read_csvs(dbms_repos_dedup_content_dir)
    local_msg_ref_columns = body_columns_dict['local_descriptions']
    local_msg_ref_columns_expanded = event_columns_dict['basic'] + body_columns_dict['local_descriptions']
    use_msg_columns = local_msg_ref_columns
    local_msg_dict = {}
    for repo_key, df_dbms_repo in df_dbms_repos_dict.items():
        local_msg_dict[repo_key] = drop_allNA(df_dbms_repo, subset=use_msg_columns, how='all',
                                              use_columns=local_msg_ref_columns_expanded)
    repo_keys = list(df_dbms_repos_dict.keys())
    # 保存实体成功匹配所过滤的substr和rawstr结果
    #   substr
    #   rawstr_filtered（仅保存all_ref_type正则匹配成功子集）
    UPDATE_REF_MSG_REGEXED_DICT_PKL = update_ref_msg_regexed_dict_pkl  # UPDATE SAVED RESULTS FLAG
    msg_substrs_filename = "repos_ref_type_local_msg_substrs_dict.pkl"
    path_repos_ref_type_local_msg_substrs_dict = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], msg_substrs_filename)
    if UPDATE_REF_MSG_REGEXED_DICT_PKL or not os.path.exists(path_repos_ref_type_local_msg_substrs_dict):
        df_repos_ref_type_local_msg_substrs_dict = find_substrs_in_df_repos_ref_type_local_msg(
            local_msg_dict, repo_keys, re_ref_patterns, use_msg_columns, record_key='id')
        dump_to_pickle(df_repos_ref_type_local_msg_substrs_dict, path_repos_ref_type_local_msg_substrs_dict,
                       update=UPDATE_REF_MSG_REGEXED_DICT_PKL)
    else:
        df_repos_ref_type_local_msg_substrs_dict = load_pickle(path_repos_ref_type_local_msg_substrs_dict)
    re_ref_types = list(re_ref_patterns.keys())
    # # just for testing: get the local message raw str of each reference type
    # df_repos_ref_type_local_msg_rawstr_dict = substrs2rawstr_in_df_repos_ref_type_local_msg(
    #     df_repos_ref_type_local_msg_substrs_dict, local_msg_dict, repo_keys, re_ref_types, use_msg_columns)
    msg_rawstr_filename = "repos_all_ref_type_local_msg_rawstr_dict.pkl"
    path_repos_ref_type_local_msg_rawstr_dict = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], msg_rawstr_filename)
    if UPDATE_REF_MSG_REGEXED_DICT_PKL or not os.path.exists(path_repos_ref_type_local_msg_rawstr_dict):
        df_repos_all_ref_type_local_msg_rawstr_dict = substrs2rawstr_in_df_repos_all_ref_type_local_msg(
            df_repos_ref_type_local_msg_substrs_dict, local_msg_dict, repo_keys, re_ref_types, use_msg_columns)
        dump_to_pickle(df_repos_all_ref_type_local_msg_rawstr_dict, path_repos_ref_type_local_msg_rawstr_dict,
                       update=UPDATE_REF_MSG_REGEXED_DICT_PKL)
    else:
        df_repos_all_ref_type_local_msg_rawstr_dict = load_pickle(path_repos_ref_type_local_msg_rawstr_dict)
    return df_repos_all_ref_type_local_msg_rawstr_dict


def stastistic_reference_substrs(use_weight_func_idx=None):
    # 统计分析
    #   item applymap:
    #     weight: len
    #     filter_count: bool
    #   column/record/ref_type/repo apply:
    #     agg_weight: sum
    #       sum_len_item
    #       sum_bool_item
    pd.set_option('display.max_columns', 10)
    # 1. statistic ref_type_local_msg_substrs
    # 1.1 calc weight for each ref type
    path_repos_ref_type_local_msg_substrs_dict = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], "repos_ref_type_local_msg_substrs_dict.pkl")
    df_repos_ref_type_local_msg_substrs_dict = load_pickle(path_repos_ref_type_local_msg_substrs_dict)
    repo_keys = list(df_repos_ref_type_local_msg_substrs_dict.keys())
    path_df_repos_ref_type_local_msg_substrscnt_map_dict_pat = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
        "repos_ref_type_local_msg_substrscnt_map_{map_cnt_func}_dict.pkl")
    path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_dict_pat = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
        "repos_ref_type_local_msg_reccnt_agg_sum_axis0_{map_cnt_func}_dict.pkl")
    path_df_repos_ref_type_local_msg_substrscnt_map_len_dict = path_df_repos_ref_type_local_msg_substrscnt_map_dict_pat.format(
        map_cnt_func='len')
    path_df_repos_ref_type_local_msg_substrscnt_map_bool_dict = path_df_repos_ref_type_local_msg_substrscnt_map_dict_pat.format(
        map_cnt_func='bool')
    path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_len_dict = path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_dict_pat.format(
        map_cnt_func='len')
    path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_bool_dict = path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_dict_pat.format(
        map_cnt_func='bool')
    WEIGHT_BY_LEN = 0
    WEIGHT_BY_BOOL = 1
    weight_func_conf = {
        WEIGHT_BY_LEN: {
            "dtype": int,
            "map_cnt_func": len,
            "path_substrscnt": path_df_repos_ref_type_local_msg_substrscnt_map_len_dict,
            "path_reccnt": path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_len_dict,
        },
        WEIGHT_BY_BOOL: {
            "dtype": bool,
            "map_cnt_func": bool,
            "path_substrscnt": path_df_repos_ref_type_local_msg_substrscnt_map_bool_dict,
            "path_reccnt": path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_bool_dict,
        }
    }
    for temp_use_weight_func_idx in weight_func_conf.keys():
        weight_func_params = weight_func_conf[temp_use_weight_func_idx]
        stat_ref_wight_df_repos_ref_type_msg(df_repos_ref_type_local_msg_substrs_dict, weight_func_params)
    # 1.2 statistic all ref type
    if use_weight_func_idx is None:
        use_weight_func_idx = WEIGHT_BY_LEN
    weight_func_params = weight_func_conf[use_weight_func_idx]
    df_repos_ref_type_local_msg_substrscnt_dict = load_pickle(weight_func_params["path_substrscnt"])
    union_ref_types = list(re_ref_patterns.keys())
    local_msg_ref_columns = body_columns_dict['local_descriptions']
    dict_repos_df_substrscnt_all_ref_type = union_all_ref_type_df_substrscnt_for_repos(
        df_repos_ref_type_local_msg_substrscnt_dict, union_ref_types=union_ref_types, subset=local_msg_ref_columns,
        dtype=weight_func_params["dtype"])
    # granularity: item match substrs len_regexed_item -> column match substrs agg_len_sum_axis1
    dict_repos_df_reccnt_agg_len_all_ref_type = agg_df_dict(dict_repos_df_substrscnt_all_ref_type, agg_func_name='sum',
                                                            subset=local_msg_ref_columns, series_as_pd_transpose=True)
    df_reccnt_agg_len_all_ref_type = concat_df_dict(dict_repos_df_reccnt_agg_len_all_ref_type, keep_k=True,
                                                    k_name='repo_key').astype(int, errors='ignore')
    print(df_reccnt_agg_len_all_ref_type.head(10))
    # granularity: record match substrs bool_regexed_item -> column match substrs agg_bool_sum_axis1
    dict_repos_df_reccnt_agg_bool_all_ref_type = agg_df_dict(
        agg_df_dict(dict_repos_df_substrscnt_all_ref_type, agg_func_name='astype', subset=local_msg_ref_columns,
                    agg_kwargs={"dtype": bool}), agg_func_name='sum')
    df_reccnt_agg_bool_all_ref_type = concat_df_dict(dict_repos_df_reccnt_agg_bool_all_ref_type, keep_k=True,
                                                     k_name='repo_key').astype(int, errors='ignore')
    print(df_reccnt_agg_bool_all_ref_type.head(10))
    # granularity: repo match substrs bool_regexed_item
    dict_repos_len_all_ref_type = agg_df_dict(dict_repos_df_substrscnt_all_ref_type, agg_func_name='__len__')
    series_repos_len_all_ref_type = pd.Series(data=dict_repos_len_all_ref_type).astype(int, errors='ignore')
    print(series_repos_len_all_ref_type.head(10))
    # 1.3 statistic each ref type
    features_freq_names = body_columns_dict['local_descriptions']
    series_feature_sum_agg = df_reccnt_agg_len_all_ref_type[features_freq_names].sum()
    df_patterns_ref_freq_cumulative = pd.DataFrame(data=series_feature_sum_agg)
    df_patterns_ref_freq_cumulative.columns = ["patterns_ref_freq"]
    df_patterns_ref_freq_cumulative['patterns_ref_log_freq'] = np.log(series_feature_sum_agg.values + 1)
    df_patterns_ref_freq_cumulative = df_patterns_ref_freq_cumulative.sort_values(by=["patterns_ref_freq"],
                                                                                  ascending=False)
    patterns_ref_freq_values = df_patterns_ref_freq_cumulative['patterns_ref_freq'].values
    cumulative_prob = [sum(patterns_ref_freq_values[:i + 1]) / sum(patterns_ref_freq_values) for i in
                       range(len(patterns_ref_freq_values))]
    df_patterns_ref_freq_cumulative['cumulative_prob'] = cumulative_prob
    return df_patterns_ref_freq_cumulative


if __name__ == '__main__':
    # query_OSDB_github_log_from_dbserver()

    # process_body_content()

    dbms_repos_dedup_content_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], 'repos_dedup_content')
    filenames = os.listdir(dbms_repos_dedup_content_dir)
    repo_keys = [os.path.splitext(full_file_name)[0] for full_file_name in filenames]
    # df_repos_all_ref_type_local_msg_rawstr_dict = identify_reference_substrs()
    # print(f"{repo_keys[0]}")
    # print(df_repos_all_ref_type_local_msg_rawstr_dict[repo_keys[0]]["all_ref_type"].head(2))

    df_patterns_ref_freq_cumulative = stastistic_reference_substrs()
    print(df_patterns_ref_freq_cumulative)
