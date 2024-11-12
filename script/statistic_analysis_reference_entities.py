#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/6/2 1:55
# @Author : 'Lou Zehua'
# @File   : statistic_analysis_reference_entities.py 

import numpy as np
import pandas as pd

from etc import filePathConf
from GH_CoRE.data_dict_settings import body_columns_dict, re_ref_patterns, USE_RAW_STR, USE_REG_SUB_STRS, \
    USE_REG_SUB_STRS_LEN, event_columns_dict
from GH_CoRE.working_flow.identify_reference import load_pickle, substrs2rawstr_in_df_repos_ref_type_local_msg, dump_to_pickle, \
    substrs2rawstr_in_df_repos_all_ref_type_local_msg, add_df_list


def substrs2substrscnt_in_df_repos_ref_type_local_msg(df_repos_ref_type_local_msg_substrs_dict, repo_keys=None,
                                                      re_ref_types=None, use_msg_columns=None, map_cnt_func=None):
    repo_keys = repo_keys or list(df_repos_ref_type_local_msg_substrs_dict.keys())
    re_ref_types = re_ref_types or list(re_ref_patterns.keys())
    use_msg_columns = use_msg_columns or body_columns_dict['local_descriptions']
    supported_cnt_func_dict = {
        "len": len,
        "bool": bool,
    }
    default_map_cnt_func = len
    msg_supported_cnt_func_list = f"supported_cnt_func_list: {list(supported_cnt_func_dict.keys())}!"
    if map_cnt_func is None:
        map_cnt_func = default_map_cnt_func
        print(f"Use default map_cunt_func: {map_cnt_func}, " + msg_supported_cnt_func_list)
    elif isinstance(map_cnt_func, str):
        try:
            map_cnt_func = supported_cnt_func_dict[map_cnt_func]
        except KeyError:
            raise KeyError(msg_supported_cnt_func_list)
    elif map_cnt_func not in supported_cnt_func_dict.values():
        print("Warning: " + msg_supported_cnt_func_list)
        pass

    df_repos_ref_type_local_msg_substrscnt_dict = {}
    for repo_key in repo_keys:
        df_local_msg_substrscnt_dict = {}
        for ref_type in re_ref_types:
            df_local_msg_substrscnt = pd.DataFrame(df_repos_ref_type_local_msg_substrs_dict[repo_key][ref_type]).copy()
            df_local_msg_substrscnt[use_msg_columns] = df_local_msg_substrscnt[use_msg_columns].applymap(
                map_cnt_func, na_action='ignore').replace(np.nan, 0).astype(int)
            df_local_msg_substrscnt_dict[ref_type] = df_local_msg_substrscnt
        df_repos_ref_type_local_msg_substrscnt_dict[repo_key] = df_local_msg_substrscnt_dict
    return df_repos_ref_type_local_msg_substrscnt_dict


def substrscnt2reccnt_in_df_repos_ref_type_local_msg(df_repos_ref_type_local_msg_substrscnt_dict, repo_keys=None,
                                                     re_ref_types=None, use_msg_columns=None, agg_colname_suffix=''):
    repo_keys = repo_keys or list(df_repos_ref_type_local_msg_substrscnt_dict.keys())
    re_ref_types = re_ref_types or list(re_ref_patterns.keys())
    use_msg_columns = use_msg_columns or body_columns_dict['local_descriptions']
    use_msg_columns_agg = use_msg_columns
    if agg_colname_suffix:
        use_msg_columns_agg = [feat + agg_colname_suffix for feat in use_msg_columns]
    df_repos_local_msg_reccnt_dict = {}
    for repo_key in repo_keys:
        df_local_msg_reccnt = pd.DataFrame(columns=["repo_key", "ref_type"] + use_msg_columns_agg)
        for ref_type in re_ref_types:
            df_local_msg_substrscnt = pd.DataFrame(df_repos_ref_type_local_msg_substrscnt_dict[repo_key][ref_type]).copy()
            df_local_msg_reccnt.loc[ref_type] = [repo_key, ref_type] + list(df_local_msg_substrscnt[use_msg_columns].sum().astype(int).values)
        df_repos_local_msg_reccnt_dict[repo_key] = df_local_msg_reccnt
    return df_repos_local_msg_reccnt_dict


def agg_df_dict(dict_k_str_v_df, subset=None, agg_func_name='sum', agg_kwargs=None, series_as_pd_transpose=False):
    subset = subset or list(dict_k_str_v_df.values())[0].columns
    agg_kwargs = agg_kwargs or {}
    if agg_func_name.lower() == 'sum':
        agg_kwargs['numeric_only'] = True
    agg_agg_NDFrame_dict = {}
    for k, df in dict_k_str_v_df.items():
        agg_func = getattr(df[subset], str(agg_func_name))
        agg_NDFrame = agg_func(**agg_kwargs)
        if isinstance(agg_NDFrame, pd.Series) and series_as_pd_transpose:
            agg_NDFrame = pd.DataFrame(agg_NDFrame).T
        agg_agg_NDFrame_dict[k] = agg_NDFrame
    return agg_agg_NDFrame_dict


def concat_df_dict(dict_k_str_v_df, reset_index=False, keep_k=True, k_name='group'):
    df_list = []
    for k, df in dict_k_str_v_df.items():
        temp_df = pd.DataFrame(df).copy()
        if isinstance(df, pd.Series):
            temp_df = temp_df.T
        use_columns = list(temp_df.columns)
        use_columns = [c for c in use_columns if c != k_name]
        use_columns = [k_name] + use_columns
        if keep_k:
            temp_df[k_name] = k
        df_list.append(temp_df[use_columns])
    concat_df = pd.concat(df_list)
    if reset_index:
        concat_df = concat_df.reset_index()
    return concat_df


# obsolete!
def stat_df_patterns_ref_freq_msg(df_repos_ref_type_local_msg_substrs_dict, repo_keys=None, re_ref_types=None,
                                  use_msg_columns=None, use_data_conf=USE_REG_SUB_STRS_LEN, raw_msg_dict=None,
                                  redundent_all_ref_type=True):
    repo_keys = repo_keys or list(df_repos_ref_type_local_msg_substrs_dict.keys())
    re_ref_types = re_ref_types or list(re_ref_patterns.keys())
    use_msg_columns = use_msg_columns or body_columns_dict['local_descriptions']
    use_data_conf_stat_func_dict = {
        USE_RAW_STR: bool,
        USE_REG_SUB_STRS: bool,
        USE_REG_SUB_STRS_LEN: len,
    }

    map_cnt_func = use_data_conf_stat_func_dict[use_data_conf]
    if use_data_conf == USE_RAW_STR:
        if raw_msg_dict is None:
            raise ValueError(f"raw_msg_dict can not be None with use_data_conf = {USE_RAW_STR}!")
        df_repos_ref_type_local_msg_rawstr_dict = substrs2rawstr_in_df_repos_ref_type_local_msg(
            df_repos_ref_type_local_msg_substrs_dict, raw_msg_dict, repo_keys, re_ref_types, use_msg_columns)
        df_repos_all_ref_type_local_msg_rawstr_dict = substrs2rawstr_in_df_repos_all_ref_type_local_msg(
            df_repos_ref_type_local_msg_substrs_dict, raw_msg_dict, repo_keys, re_ref_types, use_msg_columns)
        df_repos_msg_regexed_dict = df_repos_ref_type_local_msg_rawstr_dict if redundent_all_ref_type else df_repos_all_ref_type_local_msg_rawstr_dict
        df_repos_ref_type_local_msg_substrscnt_dict = substrs2substrscnt_in_df_repos_ref_type_local_msg(
            df_repos_msg_regexed_dict, repo_keys, re_ref_types, use_msg_columns, map_cnt_func=map_cnt_func)
    elif use_data_conf == USE_REG_SUB_STRS:
        df_repos_msg_regexed_dict = pd.DataFrame(df_repos_ref_type_local_msg_substrs_dict).copy()
        df_repos_ref_type_local_msg_substrscnt_dict = substrs2substrscnt_in_df_repos_ref_type_local_msg(
            df_repos_msg_regexed_dict, repo_keys, re_ref_types, use_msg_columns, map_cnt_func=map_cnt_func)
    else:  # USE_REG_SUB_STRS_LEN
        df_repos_ref_type_local_msg_substrscnt_dict = substrs2substrscnt_in_df_repos_ref_type_local_msg(
            df_repos_ref_type_local_msg_substrs_dict, repo_keys, re_ref_types, use_msg_columns, map_cnt_func=map_cnt_func)
        df_repos_msg_regexed_dict = df_repos_ref_type_local_msg_substrscnt_dict

    df_repos_local_msg_reccnt_dict = substrscnt2reccnt_in_df_repos_ref_type_local_msg(
        df_repos_ref_type_local_msg_substrscnt_dict, repo_keys, re_ref_types, use_msg_columns)
    df_patterns_ref_freq = df_repos_local_msg_reccnt_dict
    return df_patterns_ref_freq, df_repos_msg_regexed_dict


def stat_ref_wight_df_repos_ref_type_msg(df_repos_ref_type_local_msg_substrs_dict, weight_func_params, **kwargs):
    map_cnt_func = weight_func_params["map_cnt_func"]
    path_substrscnt = weight_func_params["path_substrscnt"]
    path_reccnt = weight_func_params["path_reccnt"]
    df_repos_ref_type_local_msg_substrscnt_dict = substrs2substrscnt_in_df_repos_ref_type_local_msg(df_repos_ref_type_local_msg_substrs_dict, map_cnt_func=map_cnt_func, **kwargs)
    dump_to_pickle(df_repos_ref_type_local_msg_substrscnt_dict, path_substrscnt)
    df_repos_local_msg_reccnt_dict = substrscnt2reccnt_in_df_repos_ref_type_local_msg(df_repos_ref_type_local_msg_substrscnt_dict, **kwargs)
    dump_to_pickle(df_repos_local_msg_reccnt_dict, path_reccnt)
    return df_repos_ref_type_local_msg_substrscnt_dict, df_repos_local_msg_reccnt_dict


def union_all_ref_type_df_substrscnt_for_repos(dict_repos_ref_type_df_substrscnt, union_ref_types=None, subset=None, dtype=int):
    subset = subset or pd.DataFrame(list(dict_repos_ref_type_df_substrscnt.values())[0]).columns
    dict_repos_df_substrscnt_all_ref_type = {}
    for repo_key, dict_ref_type_df_substrscnt in dict_repos_ref_type_df_substrscnt.items():
        if union_ref_types:
            temp_df_list = [dict_ref_type_df_substrscnt.get(key, pd.DataFrame(columns=subset)) for key in union_ref_types]
        else:
            temp_df_list = list(dict_ref_type_df_substrscnt.values())
        dict_repos_df_substrscnt_all_ref_type[repo_key] = add_df_list(temp_df_list, subset=subset, dtype=dtype)
        dict_repos_df_substrscnt_all_ref_type[repo_key] = dict_repos_df_substrscnt_all_ref_type[repo_key].astype(dtype)
    return dict_repos_df_substrscnt_all_ref_type


if __name__ == '__main__':
    import os

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

    path_df_repos_ref_type_local_msg_substrscnt_map_dict_pat = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], "repos_ref_type_local_msg_substrscnt_map_{map_cnt_func}_dict.pkl")
    path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_dict_pat = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], "repos_ref_type_local_msg_reccnt_agg_sum_axis0_{map_cnt_func}_dict.pkl")
    path_df_repos_ref_type_local_msg_substrscnt_map_len_dict = path_df_repos_ref_type_local_msg_substrscnt_map_dict_pat.format(map_cnt_func='len')
    path_df_repos_ref_type_local_msg_substrscnt_map_bool_dict = path_df_repos_ref_type_local_msg_substrscnt_map_dict_pat.format(map_cnt_func='bool')
    path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_len_dict = path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_dict_pat.format(map_cnt_func='len')
    path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_bool_dict = path_df_repos_ref_type_local_msg_reccnt_agg_sum_axis0_dict_pat.format(map_cnt_func='bool')

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
    use_weight_func_idx = WEIGHT_BY_LEN
    weight_func_params = weight_func_conf[use_weight_func_idx]
    df_repos_ref_type_local_msg_substrscnt_dict = load_pickle(weight_func_params["path_substrscnt"])
    union_ref_types = list(re_ref_patterns.keys())
    local_msg_ref_columns = body_columns_dict['local_descriptions']
    dict_repos_df_substrscnt_all_ref_type = union_all_ref_type_df_substrscnt_for_repos(
        df_repos_ref_type_local_msg_substrscnt_dict, union_ref_types=union_ref_types, subset=local_msg_ref_columns, dtype=weight_func_params["dtype"])

    # granularity: item match substrs len_regexed_item -> column match substrs agg_len_sum_axis1
    dict_repos_df_reccnt_agg_len_all_ref_type = agg_df_dict(dict_repos_df_substrscnt_all_ref_type, agg_func_name='sum', subset=local_msg_ref_columns, series_as_pd_transpose=True)
    df_reccnt_agg_len_all_ref_type = concat_df_dict(dict_repos_df_reccnt_agg_len_all_ref_type, keep_k=True, k_name='repo_key').astype(int, errors='ignore')
    print(df_reccnt_agg_len_all_ref_type.head(10))

    # granularity: record match substrs bool_regexed_item -> column match substrs agg_bool_sum_axis1
    dict_repos_df_reccnt_agg_bool_all_ref_type = agg_df_dict(agg_df_dict(dict_repos_df_substrscnt_all_ref_type, agg_func_name='astype', subset=local_msg_ref_columns, agg_kwargs={"dtype": bool}), agg_func_name='sum')
    df_reccnt_agg_bool_all_ref_type = concat_df_dict(dict_repos_df_reccnt_agg_bool_all_ref_type, keep_k=True, k_name='repo_key').astype(int, errors='ignore')
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
    print(df_patterns_ref_freq_cumulative)
