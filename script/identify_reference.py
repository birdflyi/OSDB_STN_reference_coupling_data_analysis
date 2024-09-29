#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/5/21 15:28
# @Author : 'Lou Zehua'
# @File   : identify_reference.py
import os
import sys

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = os.path.dirname(cur_dir)  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

import pickle
import re
import numpy as np
import pandas as pd

from etc import filePathConf
from script import event_columns_dict, body_columns_dict, re_ref_patterns, default_use_data_conf, USE_RAW_STR, \
    USE_REG_SUB_STRS, USE_REG_SUB_STRS_LEN, use_data_confs
from script.body_content_preprocessing import read_csvs
from script.df_sum_series_values import sum_series_values_ommit_nan


def drop_allNA(df, subset, how='all', use_columns=None):
    # preprocess nan
    df.replace('[]', np.nan, inplace=True)  # push_commits.message 是以列表存储的
    # drop na
    df = df.dropna(axis=0, how=how, subset=subset)  # 去掉全为NaN的项
    if use_columns:
        df = df[use_columns]
    return df


def test_re_ref_patterns(re_ref_patterns):
    # 测试正则
    pattern_keys = list(re_ref_patterns.keys())
    strs = [
        'href=&quot;https://github-redirect.dependabot.com/python-babel/babel/issues/782&quot;&gt;#782 once.\\n\\nRB#26080 STARTED\nBUG#32134875 "BUG#31553323" href="https://github-redirect.dependabot.com/python-babel/babel/issues/734">#734 test https://github.com/X-lab2017/open-research/issues/123#issue-1406887967 https://github.com/X-lab2017/open-digger/pull/1038#issue-1443186854 https://github.com/X-lab2017/open-galaxy/pull/2#issuecomment-982562221 https://github.com/X-lab2017/open-galaxy/pull/2#pullrequestreview-818986332 https://github.com/openframeworks/openFrameworks/pull/7383#discussion_r1411384813 https://github.com/openframeworks/openFrameworks/pull/7383/files/1f9efefc25685f062c03ebfbd2832c6e47481d01#r1411384813 https://github.com/openframeworks/openFrameworks/pull/7383/files#r1411384813 http://www.github.com/xxx/xx/issues/3221\thttps://github.com/xxx/xx/pull/3221 #3221\nissue#32 test',
        '5c9a6c1 5c9a6c2 5c9a6c12 test https://github.com/X-lab2017/open-galaxy/pull/2/commits/7f9f3706abc7b5a9ad37470519f5066119ba46c2 https://www.github.com/xxx/xx/commit/5c9a6c06871cb9fe42814af9c039eb6da5427a6e\tfile:5c9a6c06871cb9fe42814af9c039eb6da5427a6eX\n test 5c9a6c0',
        '@danxmoran1是 @danxmoran2 @danxmoran3 thank you for your help. @birdflyi是 test [birdflyi](https://github.com/birdflyi) author@abc.com\t@author test @danxmoran4) @danxmoran5',
        'test https://github.com/X-lab2017/open-research igrigorik/gharchive.org test',
        'https://github.com/birdflyi/test/tree/\'"-./()<>!%40\\nhttps://github.com/openframeworks/openFrameworks/tree/master\\n Tag: https://github.com/birdflyi/test/tree/v\'"-./()<>!%40%2540',
        'https://github.com/JuliaLang/julia/commit/5a904ac97a89506456f5e890b8dabea57bd7a0fa#commitcomment-144873925',
        'https://github.com/activescaffold/active_scaffold/wiki/API:-FieldSearch',
        '\thttps://github.com/rails/rails/releases/tag/v7.1.2\\nhttps://github.com/birdflyi/test/releases/tag/v\'"-.%2F()<>!%40%2540',
        'https://github.com/roleoroleo/yi-hack-Allwinner/files/5136276/y25ga_0.1.9.tar.gz是 https://github.com/X-lab2017/open-digger/pull/997/files#diff-5cda5bb2aa8682c3b9d4dbf864efdd6100fe1a5f748941d972204412520724e5 https://github.com/birdflyi/Research-Methods-of-Cross-Science/blob/main/%E4%BB%8E%E7%A7%91%E5%AD%A6%E8%B5%B7%E6%BA%90%E7%9C%8B%E4%BA%A4%E5%8F%89%E5%AD%A6%E7%A7%91.md是',
        'test https://github.com/X-lab2017/open-digger/labels/pull%2Fhypertrons test',
        'https://gist.github.com/birdflyi',
        'test http://sqlite.org/forum/forumpost/fdb0bb7ad0 https://sqlite.org/forum/forumpost/fdb0bb7ad0\n\nhttps://github.com\thttps://www.github.com test'
        ]

    CORRESPONDING = 0
    MIXED = 1
    test_mode = MIXED

    for i in range(len(pattern_keys)):
        pattern_key = pattern_keys[i]
        s = ' '.join(strs) if test_mode == MIXED else strs[i]
        s_idx = 'all' if test_mode == MIXED else str(i)
        for j in range(len(re_ref_patterns[pattern_key])):
            pat = re_ref_patterns[pattern_key][j]
            regex = re.compile(pat)
            subs = re.findall(regex, s)  # match总是从string的开始位置匹配，search不能找到所有括号的内容，应该使用findall
            print(f"{pattern_key}_{j} strs_{s_idx} subs: {subs}")


def strs_regex(strs, regex):
    find_res_lists = []
    for s in strs:
        subs = re.findall(regex, s)  # match总是从string的开始位置匹配，search不能找到所有括号的内容，应该使用findall
        find_res_lists.append(subs)
    return find_res_lists


def mask_code(string):
    # 正则表达式来匹配代码块（用```括起来的）
    code_block_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    # 正则表达式来匹配行内代码（用`括起来的）
    inline_code_pattern = re.compile(r'`([^`]+)`')
    # 替换代码块
    string_mask_code__block = re.sub(code_block_pattern, "CODE_BLOCK", string)
    # 替换行内代码
    string_mask_code = re.sub(inline_code_pattern, "INLINE_CODE", string_mask_code__block)
    return string_mask_code

def regex_df(df, regex_columns, regex_pattern, use_data_conf=None):
    if use_data_conf is None:
        use_data_conf = default_use_data_conf  # granularity configuration
    # print(df)
    regex = re.compile(regex_pattern)
    df_regexed = pd.DataFrame()

    for column in regex_columns:
        series = df[column].astype(str)
        find_res_lists = strs_regex(series.apply(mask_code).values, regex)
        regexed_series = pd.Series(data=find_res_lists, index=series.index, dtype=object)
        regexed_len_series = regexed_series.apply(len)
        if use_data_conf == USE_RAW_STR:
            temp_series = series
        elif use_data_conf == USE_REG_SUB_STRS:
            temp_series = regexed_series
        elif use_data_conf == USE_REG_SUB_STRS_LEN:
            temp_series = regexed_len_series
        else:
            print('use_data_conf must be in use_data_confs:', use_data_confs)
            return
        df_regexed[column] = pd.Series(np.where(regexed_len_series.values > 0, temp_series.values, np.nan), index=temp_series.index) # 将匹配到子串的结果长度为0时设置为np.nan是为下一步过滤做准备，特别是出现'[]'时应被替换为np.nan，以被update更新

    df_regexed_fullcols = df.copy()
    df_regexed_fullcols[regex_columns] = np.nan
    df_regexed_fullcols.update(df_regexed, overwrite=True)
    return df_regexed_fullcols


# df cancat, push the same index-column items into a list and keep the first record grouped by share columns, ommit nan.
# # 处理结果：
# key_col + share_cols + union_cols
# 其中：
#     key_col唯一去重
#     share_cols取按key_col聚集的group by的first
#     union_cols取按key_col聚集的group by的全体元素列表
def df_union_agg_sumlist(df_list, on_col, union_cols, share_cols=None, merge_how__share_union='left',
                         keep_notalllist_notallnum_str='first'):
    df_raw_cancat = pd.concat(list(df_list))
    # print('df_raw_cancat', df_raw_cancat)
    DEFAULT_INDEX_NAME = 'index'
    DF_INDEX_HAS_NAME = df_raw_cancat.index.name != None
    if not DF_INDEX_HAS_NAME:
        df_raw_cancat.index.name = DEFAULT_INDEX_NAME

    raw_index_name = df_raw_cancat.index.name
    df_raw_cancat.reset_index(inplace=True)  # 将index列连同列名转为普通列
    if not on_col:
        print("union on_col is None, try to use df.index.name as default union on_col!")

    key_col = on_col or DEFAULT_INDEX_NAME
    df_merge = pd.DataFrame(columns=df_raw_cancat.columns)
    df_raw_cancat = df_raw_cancat.dropna(axis=0, how='all', subset=union_cols).dropna(axis=0, how='any', subset=key_col)
    if df_raw_cancat.empty:
        df_merge.set_index(raw_index_name, inplace=True)
        if not DF_INDEX_HAS_NAME:  # 恢复原始index命名
            df_merge.index.name = None
        return df_merge
    gb = df_raw_cancat.groupby(by=key_col)

    if key_col in union_cols:
        print("Error: union on_col = {key_col} should not be included in union_cols = {union_cols}!".format(
            key_col=key_col, union_cols=union_cols))
        return

    share_cols = share_cols or [c for c in df_raw_cancat.columns if c not in union_cols]
    if key_col not in share_cols:
        share_cols = [key_col] + share_cols

    df_union_groupby_dict = {}
    for idx, i_df in gb:
        # print('i_df', i_df[union_cols])
        df_union = pd.DataFrame(i_df[union_cols].apply(sum_series_values_ommit_nan, keep_notalllist_notallnum_str=keep_notalllist_notallnum_str, axis=0))
        df_union_groupby_dict[idx] = df_union

    df_union_cols = pd.concat(list(df_union_groupby_dict.values()))
    df_union_cols[key_col] = list(df_union_groupby_dict.keys())

    # print('df_union_cols', df_union_cols)
    df_share_cols = df_raw_cancat[share_cols].groupby(by=key_col, as_index=False).first()  # share_cols列，按key_col分组，每组取第一个
    assert (len(df_share_cols) == len(df_union_cols))
    df_merge = pd.merge(df_share_cols, df_union_cols, on=key_col, how=merge_how__share_union)
    df_merge.set_index(raw_index_name, inplace=True)
    if not DF_INDEX_HAS_NAME:  # 恢复原始index命名
        df_merge.index.name = None
    return df_merge


def df_regexed_union_by_patterns(df, use_msg_columns, regex_pattern_list, use_data_conf=None, union_on_col=None):
    # use_msg_columns = ['issue_body']  # just for debug
    df_regexed_list = []
    for regex_pattern in regex_pattern_list:
        temp_df = regex_df(df, use_msg_columns, regex_pattern, use_data_conf=use_data_conf)
        # temp_df.replace(False, np.nan, inplace=True)
        df_regexed_list.append(temp_df)
        # df_regexed_list.append(temp_df.loc[181:290])  # just for test "mysql_mysql-server" 'local' 'issue_body'

    df_regexed_union = df_union_agg_sumlist(df_regexed_list, on_col=union_on_col, union_cols=use_msg_columns, keep_notalllist_notallnum_str='first')
    df_regexed_union = df_regexed_union.dropna(axis=0, how='all', subset=use_msg_columns)
    df_regexed_union.sort_index(ascending=True, inplace=True)
    if use_data_conf == USE_REG_SUB_STRS_LEN:
        try:
            df_regexed_union[use_msg_columns] = df_regexed_union[use_msg_columns].replace(np.nan, 0)
            df_regexed_union[use_msg_columns] = df_regexed_union[use_msg_columns].astype(int)
        except ValueError:
            print('Warning: dtype auto changed! Change the column dtype from int to float while use_data_conf == USE_REG_SUB_STRS_LEN!')
    return df_regexed_union


def test_df_regexed_union_by_patterns(df_local_msg):
    use_data_conf = 0
    regex_patterns_keyname = 'Issue_PR'
    regex_patterns = re_ref_patterns[regex_patterns_keyname]

    pd.set_option('display.max_columns', 10)
    df_local_msg_regexed_dict = {}
    for ref_type, regex_pattern in enumerate(regex_patterns):
        # use_msg_columns = ['issue_body']
        use_msg_columns = body_columns_dict['local_descriptions']
        df_local_msg_regexed = df_regexed_union_by_patterns(df_local_msg, use_msg_columns, regex_patterns, use_data_conf=use_data_conf, union_on_col='id')
        df_local_msg_regexed_dict[ref_type] = df_local_msg_regexed
        print('--', regex_patterns_keyname, '-', ref_type, '-', use_msg_columns)
        break
    print(df_local_msg_regexed_dict)
    return df_local_msg_regexed_dict


# obsoleted! 对所有的pattern统计交叉引用情况
def get_df_patterns_ref_freq_msg(repo_keys, re_ref_patterns, local_msg_dict, use_msg_columns=None, use_data_conf=default_use_data_conf):
    df_repos_msg_regexed_dict = {}
    use_msg_columns = use_msg_columns or body_columns_dict['local_descriptions']
    df_repos_reftype_findfreq_columns = ["repo_key", "ref_type"] + [feat + '_freq' for feat in use_msg_columns]
    df_patterns_ref_freq = pd.DataFrame(columns=df_repos_reftype_findfreq_columns)
    temp_data_matrix = []
    for repo_key in repo_keys:
        df_local_msg_regexed_dict = {}
        for ref_type, regex_patterns in re_ref_patterns.items():
            df_local_msg_regexed = df_regexed_union_by_patterns(local_msg_dict[repo_key], use_msg_columns, regex_patterns, use_data_conf=use_data_conf, union_on_col='id')
            df_local_msg_regexed_dict[ref_type] = df_local_msg_regexed
            if use_data_conf == USE_REG_SUB_STRS_LEN:
                temp_data_matrix.append([repo_key, ref_type] + list(df_local_msg_regexed[use_msg_columns].sum().astype(int).values))
            else:
                temp_data_matrix.append([repo_key, ref_type] + list(df_local_msg_regexed[use_msg_columns].count().astype(int).values))
        df_repos_msg_regexed_dict[repo_key] = df_local_msg_regexed_dict
    df_patterns_ref_freq = pd.DataFrame(data=temp_data_matrix, columns=df_patterns_ref_freq.columns)
    return df_patterns_ref_freq, df_repos_msg_regexed_dict


def find_substrs_in_df_repos_ref_type_local_msg(df_repos_local_msg_rawstr_dict, repo_keys=None, ref_type_regex_patterns_dict=None,
                                                use_msg_columns=None, record_key='id'):
    repo_keys = repo_keys or list(df_repos_local_msg_rawstr_dict.keys())
    ref_type_regex_patterns_dict = ref_type_regex_patterns_dict or dict(re_ref_patterns)
    use_msg_columns = use_msg_columns or body_columns_dict['local_descriptions']
    df_repos_ref_type_local_msg_substrs_dict = {}
    for repo_key in repo_keys:
        df_local_msg_regexed_dict = {}
        df_local_msg = df_repos_local_msg_rawstr_dict[repo_key]
        for ref_type, regex_patterns in ref_type_regex_patterns_dict.items():
            df_local_msg_regexed = df_regexed_union_by_patterns(df_local_msg, use_msg_columns, regex_patterns, use_data_conf=USE_REG_SUB_STRS, union_on_col=record_key)
            df_local_msg_regexed_dict[ref_type] = df_local_msg_regexed
        df_repos_ref_type_local_msg_substrs_dict[repo_key] = df_local_msg_regexed_dict
    return df_repos_ref_type_local_msg_substrs_dict


# To get the local message raw str of each reference type as a runtime view.
# Statistical Warning: There may be multi-types matches in one str.
def substrs2rawstr_in_df_repos_ref_type_local_msg(df_repos_ref_type_local_msg_substrs_dict, local_msg_dict,
                                                  repo_keys=None, re_ref_types=None, use_msg_columns=None,
                                                  keep_all_col=True):
    repo_keys = repo_keys or list(df_repos_ref_type_local_msg_substrs_dict.keys())
    re_ref_types = re_ref_types or list(re_ref_patterns.keys())
    use_msg_columns = use_msg_columns or body_columns_dict['local_descriptions']
    filter_columns = use_msg_columns

    df_repos_ref_type_local_msg_rawstr_dict = {}
    for repo_key in repo_keys:
        df_local_msg_rawstr_regfiltered_dict = {}
        df_local_msg = local_msg_dict[repo_key]
        result_columns = df_local_msg.columns if keep_all_col else filter_columns
        remain_columns = [c for c in result_columns if c not in filter_columns]
        for ref_type in re_ref_types:
            df_local_msg_substrs = pd.DataFrame(df_repos_ref_type_local_msg_substrs_dict[repo_key][ref_type]).copy()
            df_local_msg_substrs_filter = df_local_msg_substrs[use_msg_columns].applymap(
                bool, na_action='ignore').replace(np.nan, False).astype(bool)
            df_local_msg_substrs_filter[remain_columns] = True
            df_local_msg_rawstr_regfiltered = df_local_msg.loc[df_local_msg_substrs_filter.index, result_columns][df_local_msg_substrs_filter]
            df_local_msg_rawstr_regfiltered_dict[ref_type] = df_local_msg_rawstr_regfiltered
        df_repos_ref_type_local_msg_rawstr_dict[repo_key] = df_local_msg_rawstr_regfiltered_dict
    return df_repos_ref_type_local_msg_rawstr_dict


def add_df_list(df_list, subset=None, fill_value=0, dtype=float):
    if not len(df_list):
        return None
    subset = subset or df_list[0].columns
    df_add = pd.DataFrame(columns=subset)
    for df in df_list:
        df_add = df_add.add(df[subset].astype(dtype), fill_value=fill_value).astype(dtype)
    return df_add


def get_df_bool_mask(df):
    return df.replace(np.nan, 0).astype(bool)


# To save the local message raw str which exists any reference type.
def substrs2rawstr_in_df_repos_all_ref_type_local_msg(df_repos_ref_type_local_msg_substrs_dict, local_msg_dict,
                                                      repo_keys=None, re_ref_types=None, use_msg_columns=None,
                                                      keep_all_col=True):
    repo_keys = repo_keys or list(df_repos_ref_type_local_msg_substrs_dict.keys())
    re_ref_types = re_ref_types or list(re_ref_patterns.keys())
    use_msg_columns = use_msg_columns or body_columns_dict['local_descriptions']
    filter_columns = use_msg_columns

    df_repos_all_ref_type_local_msg_rawstr_dict = {}
    for repo_key in repo_keys:
        df_local_msg_rawstr_regfiltered_dict = {}
        df_local_msg = local_msg_dict[repo_key]
        result_columns = df_local_msg.columns if keep_all_col else filter_columns
        remain_columns = [c for c in result_columns if c not in filter_columns]
        df_local_msg_substrs_filters = []
        for ref_type in re_ref_types:
            df_local_msg_substrs = pd.DataFrame(df_repos_ref_type_local_msg_substrs_dict[repo_key][ref_type]).copy()
            df_local_msg_substrs_filter = df_local_msg_substrs[filter_columns].applymap(
                bool, na_action='ignore').replace(np.nan, False).astype(bool)
            df_local_msg_substrs_filters.append(df_local_msg_substrs_filter)
        df_local_msg_all_ref_type_filter = get_df_bool_mask(add_df_list(df_local_msg_substrs_filters, subset=filter_columns, dtype=int))
        df_local_msg_all_ref_type_filter[remain_columns] = True
        df_local_msg_rawstr_regfiltered = df_local_msg.loc[df_local_msg_all_ref_type_filter.index, result_columns][df_local_msg_all_ref_type_filter]
        df_local_msg_rawstr_regfiltered_dict["all_ref_type"] = df_local_msg_rawstr_regfiltered
        df_repos_all_ref_type_local_msg_rawstr_dict[repo_key] = df_local_msg_rawstr_regfiltered_dict
    return df_repos_all_ref_type_local_msg_rawstr_dict


def dump_to_pickle(d, save_path, update=True):
    filename = os.path.basename(save_path)
    if update or not os.path.exists(save_path):
        with open(save_path, "wb") as fp:
            pickle.dump(d, fp)
        print(f"{filename} saved!")
    else:
        print(f"{filename} exists!")


def load_pickle(load_path):
    with open(load_path, "rb") as fp:
        load_dict = pickle.load(fp)
    return load_dict


if __name__ == '__main__':
    # 读入csv，筛选项目
    dbms_repos_dedup_content_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], 'repos_dedup_content')
    df_dbms_repos_dict = read_csvs(dbms_repos_dedup_content_dir)

    local_msg_ref_columns = body_columns_dict['local_descriptions']
    local_msg_ref_columns_expanded = event_columns_dict['basic'] + body_columns_dict['local_descriptions']
    use_msg_columns = local_msg_ref_columns
    local_msg_dict = {}
    for repo_key, df_dbms_repo in df_dbms_repos_dict.items():
        local_msg_dict[repo_key] = drop_allNA(df_dbms_repo, subset=use_msg_columns, how='all', use_columns=local_msg_ref_columns_expanded)

    # test_re_ref_patterns(re_ref_patterns)
    repo_keys = list(df_dbms_repos_dict.keys())
    # for test
    # repo_keys = repo_keys[:10]
    # repo_keys = ["basho_riak_kv_2022"]
    # test_df_regexed_union_by_patterns(local_msg_dict[repo_keys[0]])

    # # obsoleted! test for the consistency of get_df_patterns_ref_freq returns and saved data
    # udc = use_data_confs[2]
    # df_patterns_ref_freq, df_repos_msg_regexed_dict = get_df_patterns_ref_freq_msg(repo_keys, re_ref_patterns, local_msg_dict, use_msg_columns, use_data_conf=udc)

    # 保存实体成功匹配所过滤的substr和rawstr结果
    #   substr
    #   rawstr_filtered（仅保存all_ref_type正则匹配成功子集）
    UPDATE_REF_MSG_REGEXED_DICT_PKL = False  # UPDATE SAVED RESULTS FLAG

    msg_substrs_filename = "repos_ref_type_local_msg_substrs_dict.pkl"
    path_repos_ref_type_local_msg_substrs_dict = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], msg_substrs_filename)
    if UPDATE_REF_MSG_REGEXED_DICT_PKL or not os.path.exists(path_repos_ref_type_local_msg_substrs_dict):
        df_repos_ref_type_local_msg_substrs_dict = find_substrs_in_df_repos_ref_type_local_msg(
            local_msg_dict, repo_keys, re_ref_patterns, use_msg_columns, record_key='id')
        dump_to_pickle(df_repos_ref_type_local_msg_substrs_dict, path_repos_ref_type_local_msg_substrs_dict, update=UPDATE_REF_MSG_REGEXED_DICT_PKL)
    else:
        df_repos_ref_type_local_msg_substrs_dict = load_pickle(path_repos_ref_type_local_msg_substrs_dict)

    re_ref_types = list(re_ref_patterns.keys())

    # # just for testing: get the local message raw str of each reference type
    df_repos_ref_type_local_msg_rawstr_dict = substrs2rawstr_in_df_repos_ref_type_local_msg(
        df_repos_ref_type_local_msg_substrs_dict, local_msg_dict, repo_keys, re_ref_types, use_msg_columns)

    msg_rawstr_filename = "repos_all_ref_type_local_msg_rawstr_dict.pkl"
    path_repos_ref_type_local_msg_rawstr_dict = os.path.join(
        filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], msg_rawstr_filename)
    if UPDATE_REF_MSG_REGEXED_DICT_PKL or not os.path.exists(path_repos_ref_type_local_msg_rawstr_dict):
        df_repos_all_ref_type_local_msg_rawstr_dict = substrs2rawstr_in_df_repos_all_ref_type_local_msg(
            df_repos_ref_type_local_msg_substrs_dict, local_msg_dict, repo_keys, re_ref_types, use_msg_columns)
        dump_to_pickle(df_repos_all_ref_type_local_msg_rawstr_dict, path_repos_ref_type_local_msg_rawstr_dict, update=UPDATE_REF_MSG_REGEXED_DICT_PKL)
    else:
        df_repos_all_ref_type_local_msg_rawstr_dict = load_pickle(path_repos_ref_type_local_msg_rawstr_dict)
    # test load pickle
    print(f"{repo_keys[0]}")
    print(df_repos_all_ref_type_local_msg_rawstr_dict[repo_keys[0]]["all_ref_type"].head(5))
