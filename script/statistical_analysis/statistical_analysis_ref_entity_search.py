#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/14 5:47
# @Author : 'Lou Zehua'
# @File   : statistical_analysis_ref_entity_search.py

import ast
import logging
import os

import numpy as np
import pandas as pd

from etc import filePathConf, pkg_rootdir
from GH_CoRE.data_dict_settings import columns_simple, body_columns_dict, event_columns_dict, re_ref_patterns
from GH_CoRE.model import ObjEntity
from GH_CoRE.working_flow.body_content_preprocessing import read_csvs
from GH_CoRE.working_flow.identify_reference import drop_allNA, find_substrs_in_df_repos_ref_type_local_msg, dump_to_pickle, \
    load_pickle, substrs2rawstr_in_df_repos_all_ref_type_local_msg
from GH_CoRE.working_flow.query_OSDB_github_log import query_repo_log_each_year_to_csv_dir
from GH_CoRE.utils.logUtils import setup_logging

from script.build_dataset.collaboration_relation_extraction import process_body_content, \
    collaboration_relation_extraction
from script.build_dataset.query_repos_event_log import query_OSDB_github_log_from_dbserver
from script.build_dataset.repo_filter import get_filenames_by_repo_names, get_intersection
from script.statistical_analysis.statistical_analysis_ref_named_entity import stat_ref_wight_df_repos_ref_type_msg, \
    union_all_ref_type_df_substrscnt_for_repos, agg_df_dict, concat_df_dict

setup_logging(base_dir=pkg_rootdir)
logger = logging.getLogger(__name__)


def identify_reference_substrs(df_dbms_repos_dict, ref_substrs_pkl_save_path=None, ref_rawstr_pkl_save_path=None,
                               update_ref_msg_regexed_dict_pkl=False, **kwargs):
    if not update_ref_msg_regexed_dict_pkl:
        return

    local_msg_ref_columns = body_columns_dict['local_descriptions']
    local_msg_ref_columns_expanded = event_columns_dict['basic'] + body_columns_dict['local_descriptions']
    use_msg_columns = local_msg_ref_columns
    local_msg_dict = {}
    use_repo_keys = kwargs.get("use_repo_keys", None)
    stop_repo_keys = kwargs.get("stop_repo_keys", None)
    flag_custom_keys = bool(use_repo_keys or stop_repo_keys)

    use_repo_keys = use_repo_keys or list(df_dbms_repos_dict.keys())
    stop_repo_keys = stop_repo_keys or []
    for repo_key, df_dbms_repo in df_dbms_repos_dict.items():
        if repo_key in use_repo_keys and repo_key not in stop_repo_keys:
            local_msg_dict[repo_key] = drop_allNA(df_dbms_repo, subset=use_msg_columns, how='all',
                                                  use_columns=local_msg_ref_columns_expanded)
    repo_keys = list(local_msg_dict.keys())
    if flag_custom_keys:
        print("The custom keys: {repo_keys}".format(repo_keys=repo_keys))

    # 保存实体成功匹配所过滤的substr和rawstr结果
    #   substr 命名实体
    #   rawstr_filtered 含命名实体的原始数据项（仅保存all_ref_type正则匹配成功子集）
    UPDATE_REF_MSG_REGEXED_DICT_PKL = update_ref_msg_regexed_dict_pkl  # UPDATE SAVED RESULTS FLAG
    msg_substrs_filename = "repos_ref_type_local_msg_substrs_dict.pkl"
    ref_substrs_pkl_save_path = ref_substrs_pkl_save_path or os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], msg_substrs_filename)
    if UPDATE_REF_MSG_REGEXED_DICT_PKL or not os.path.exists(ref_substrs_pkl_save_path):
        df_repos_ref_type_local_msg_substrs_dict = find_substrs_in_df_repos_ref_type_local_msg(
            local_msg_dict, repo_keys, re_ref_patterns, use_msg_columns, record_key='id')
        dump_to_pickle(df_repos_ref_type_local_msg_substrs_dict, ref_substrs_pkl_save_path,
                       update=UPDATE_REF_MSG_REGEXED_DICT_PKL)
    else:
        df_repos_ref_type_local_msg_substrs_dict = load_pickle(ref_substrs_pkl_save_path)
    re_ref_types = list(re_ref_patterns.keys())
    # # just for testing: get the local message raw str of each reference type
    # df_repos_ref_type_local_msg_rawstr_dict = substrs2rawstr_in_df_repos_ref_type_local_msg(
    #     df_repos_ref_type_local_msg_substrs_dict, local_msg_dict, repo_keys, re_ref_types, use_msg_columns)
    msg_rawstr_filename = "repos_all_ref_type_local_msg_rawstr_dict.pkl"
    ref_rawstr_pkl_save_path = ref_rawstr_pkl_save_path or os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], msg_rawstr_filename)
    if UPDATE_REF_MSG_REGEXED_DICT_PKL or not os.path.exists(ref_rawstr_pkl_save_path):
        df_repos_all_ref_type_local_msg_rawstr_dict = substrs2rawstr_in_df_repos_all_ref_type_local_msg(
            df_repos_ref_type_local_msg_substrs_dict, local_msg_dict, repo_keys, re_ref_types, use_msg_columns)
        dump_to_pickle(df_repos_all_ref_type_local_msg_rawstr_dict, ref_rawstr_pkl_save_path,
                       update=UPDATE_REF_MSG_REGEXED_DICT_PKL)
    else:
        # df_repos_all_ref_type_local_msg_rawstr_dict = load_pickle(ref_rawstr_pkl_save_path)
        pass
    return None


def stastistic_reference_substrs(ref_substrs_pkl_path=None, use_weight_func_idx=None):
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
    ref_substrs_pkl_path = ref_substrs_pkl_path or path_repos_ref_type_local_msg_substrs_dict
    df_repos_ref_type_local_msg_substrs_dict = load_pickle(ref_substrs_pkl_path)
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
    df_patterns_ref_freq_cumulative = get_df_feat_statistic(series_feature_sum_agg)
    return df_patterns_ref_freq_cumulative


def get_df_feat_statistic(series_feature_sum_agg):
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


def roughly_valid_reference_entity(rec):
    is_reference_entity = rec is not None and rec["relation_type"] == "Reference"
    if not is_reference_entity:
        return False

    tar_entity_objnt_prop_dict = rec["tar_entity_objnt_prop_dict"]
    tar_entity_objnt_prop_dict = ast.literal_eval(tar_entity_objnt_prop_dict) if pd.notna(tar_entity_objnt_prop_dict) else {}
    roughly_valid = rec["tar_entity_type"] != "Obj" or tar_entity_objnt_prop_dict.get("label", "NotAnEntity") != "NotAnEntity"
    return is_reference_entity and roughly_valid


def get_grouped_ref_ent_cnt(df: pd.DataFrame, groupby_filed):
    cnt_field = "is_roughly_valid_ref_entity"
    df[cnt_field] = df.apply(roughly_valid_reference_entity, axis=1).astype(int)
    df_group_sum = df.groupby(groupby_filed)[cnt_field].sum().reset_index()
    series_feature_sum_agg = df_group_sum.set_index(groupby_filed)[cnt_field]
    return series_feature_sum_agg


if __name__ == '__main__':
    year = 2023
    repo_names = ["TuGraph-family/tugraph-db", "facebook/rocksdb", "cockroachdb/cockroach"][0:2]
    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]
    dbms_repos_dedup_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_DEDUP_CONTENT_DIR]

    if repo_names:
        sql_param = {
            "table": "opensource.events",
            "start_end_year": [year, year + 1],
        }
        query_repo_log_each_year_to_csv_dir(repo_names, columns=columns_simple, save_dir=dbms_repos_raw_content_dir,
                                            sql_param=sql_param)
    else:
        query_OSDB_github_log_from_dbserver(key_feats_path=dbms_repos_key_feats_path, save_dir=dbms_repos_raw_content_dir)

    # Filter files
    df_OSDB_github_key_feats = pd.read_csv(dbms_repos_key_feats_path, header='infer', index_col=None)
    df_OSDB_github_key_feats_has_github_repo_id = df_OSDB_github_key_feats[
        pd.notna(df_OSDB_github_key_feats["github_repo_id"])]  # filter github_repo_id must exist
    repo_names_has_github_repo_id = list(df_OSDB_github_key_feats_has_github_repo_id["github_repo_link"].values)
    filenames_has_github_repo_id = get_filenames_by_repo_names(repo_names_has_github_repo_id, year)
    filenames_exists = os.listdir(dbms_repos_raw_content_dir)
    if repo_names is not None:
        filenames_used = get_filenames_by_repo_names(repo_names, year)
    else:
        filenames_used = filenames_exists
    filenames_scope_list = [filenames_used, filenames_exists, filenames_has_github_repo_id]
    filenames = get_intersection(filenames_scope_list)
    logger.log(logging.INFO, msg=f"Use {len(filenames)} filenames: {filenames}")

    # Preprocess body content
    process_body_content(raw_content_dir=dbms_repos_raw_content_dir, processed_content_dir=dbms_repos_dedup_content_dir, filenames=filenames)
    df_dbms_repos_dict = read_csvs(dbms_repos_dedup_content_dir, filenames=filenames, index_col=0)
    repo_keys = list(df_dbms_repos_dict.keys())

    # Statistics: Fields containing reference entities
    msg_substrs_filename = "repos_ref_type_local_msg_substrs_dict.pkl"
    path_repos_ref_type_local_msg_substrs_dict = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], msg_substrs_filename)
    msg_rawstr_filename = "repos_all_ref_type_local_msg_rawstr_dict.pkl"
    path_repos_ref_type_local_msg_rawstr_dict = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], msg_rawstr_filename)

    update_ref_msg_regexed_dict_pkl = False
    identify_reference_substrs(df_dbms_repos_dict, ref_substrs_pkl_save_path=path_repos_ref_type_local_msg_substrs_dict,
        ref_rawstr_pkl_save_path=path_repos_ref_type_local_msg_rawstr_dict, use_repo_keys=repo_keys,
        update_ref_msg_regexed_dict_pkl=update_ref_msg_regexed_dict_pkl)

    df_repos_all_ref_type_local_msg_rawstr_dict = load_pickle(path_repos_ref_type_local_msg_rawstr_dict)
    pd.set_option('display.max_columns', None)
    print(df_repos_all_ref_type_local_msg_rawstr_dict[repo_keys[0]]["all_ref_type"].head(2))

    WEIGHT_BY_LEN = 0
    WEIGHT_BY_BOOL = 1
    df_patterns_ref_freq_cumulative = stastistic_reference_substrs(
        ref_substrs_pkl_path=path_repos_ref_type_local_msg_substrs_dict, use_weight_func_idx=WEIGHT_BY_LEN)
    print(df_patterns_ref_freq_cumulative)

    # Statistics: Types of reference entities
    relation_extraction_save_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], "repos_GH_CoRE")
    collaboration_relation_extraction(repo_keys, df_dbms_repos_dict, relation_extraction_save_dir, update_exists=False, add_mode_if_exists=True)
    df_relation_extraction_dict = read_csvs(relation_extraction_save_dir, filenames=filenames)

    repos_all_ref_type_ref_ent_cnt_dict_filename = "repos_all_ref_type_ref_ent_cnt_dict.pkl"
    repos_all_ref_type_ref_ent_cnt_dict_path = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], repos_all_ref_type_ref_ent_cnt_dict_filename)

    update_all_ref_type_ref_ent_cnt_dict_pkl = False
    if update_all_ref_type_ref_ent_cnt_dict_pkl:
        ser_repos_all_ref_type_ref_ent_cnt_dict = {k: get_grouped_ref_ent_cnt(v_df, groupby_filed="tar_entity_type") for k, v_df in df_relation_extraction_dict.items()}
        dump_to_pickle(ser_repos_all_ref_type_ref_ent_cnt_dict, repos_all_ref_type_ref_ent_cnt_dict_path, update=True)

    ser_repos_all_ref_type_ref_ent_cnt_dict = load_pickle(repos_all_ref_type_ref_ent_cnt_dict_path)
    series_repos_all_ref_type_ref_ent_cnt = pd.Series(ser_repos_all_ref_type_ref_ent_cnt_dict[repo_keys[0]])
    all_ref_types = ObjEntity.E.keys()
    for name in all_ref_types:
        if name not in series_repos_all_ref_type_ref_ent_cnt.keys():
            series_repos_all_ref_type_ref_ent_cnt[name] = 0
    df_patterns_ref_freq_cumulative = get_df_feat_statistic(series_repos_all_ref_type_ref_ent_cnt)
    print(df_patterns_ref_freq_cumulative)
