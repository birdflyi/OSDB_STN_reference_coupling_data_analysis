#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/5/23 5:40
# @Author : 'Lou Zehua'
# @File   : df_sum_series_values.py 

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

import numpy as np
import pandas as pd
from utils.check_type import is_nan, is_number, is_list, is_str


def sum_nums_or_lists(lists):
    list_str_fromat = lambda x: type(x) == str and x.startswith('[') and x.endswith(']')

    try:
        if list_str_fromat(lists):
            lists = eval(lists)

        res = sum(lists)
    except BaseException:
        # only for str(list[list]) type data
        res = []
        for l in lists:
            if list_str_fromat(l):  # list_str_fromat('NaN') == False, will be filtered here.
                l = eval(l)
            if type(l) == list:
                res += l
            else:
                return np.nan
        # res = str(res)  # 必要，series.apply()时，sum of lists得到的结果是list，需要保持转series后仍是一个整体
    return res


def sum_series_values_ommit_nan(series, keep_notalllist_notallnum_str='first'):
    nan_series_value = np.nan
    nan_series = pd.Series([nan_series_value])
    series = pd.Series(series)
    ser_vals = list(series.values)

    flag_alllist_or_allnum = False
    try:
        flag_alllist = all([is_list(x, True) for x in ser_vals])
        flag_allnum = all([is_number(x, True) for x in ser_vals])
        flag_alllist_or_allnum = flag_alllist or flag_allnum
    except ValueError:
        pass
    flag_allstr = False
    try:
        flag_allstr = all([is_str(x, True) for x in ser_vals])
    except ValueError:
        pass

    ser_vals_filtered = [i for i in ser_vals if not is_nan(str(i))]
    if flag_alllist_or_allnum:
        res_series_value = sum_nums_or_lists(ser_vals_filtered) if len(ser_vals_filtered) else nan_series_value
        if is_nan(res_series_value):
            res_series_value = nan_series_value
    elif flag_allstr:
        if not len(ser_vals_filtered):
            return nan_series

        if keep_notalllist_notallnum_str is False:
            return nan_series
        elif keep_notalllist_notallnum_str == 'first':
            return pd.Series([ser_vals_filtered[0]])
        elif keep_notalllist_notallnum_str == 'last':
            return pd.Series([ser_vals_filtered[-1]])
        elif keep_notalllist_notallnum_str == 'queue':
            temp_ser_vals_filtered = []
            for it in ser_vals_filtered:
                temp_ser_vals_filtered += it if type(it) is list else [it]
            return pd.Series([temp_ser_vals_filtered])
        else:
            print("Error: keep_notlist_notnum_str must be set in [False, 'first', 'last', 'queue']!")
            return None
    else:
        print("Warning: Unhandled type! return np.nan!")
        return nan_series
    res_series = pd.Series([res_series_value])
    return res_series


def test_sum_nums_or_lists():
    lists = [[1, 11], [2], [3], [4]]
    nums = [1, 2, 3]
    print("sum lists:", sum_nums_or_lists(lists))
    print("sum nums:", sum_nums_or_lists(nums))
    return


def test_sum_series_values_ommit_nan(keep_notalllist_notallnum_str='first'):
    # test for dataframe apply function series_lists_sum_ommit_nan
    df_temp = pd.DataFrame({'issue_body': [["a", "aa"], ["b"]], 'issue_body2': [['a'], np.nan], 'x': ['a', 'b'],
                           'xx': ['a', np.nan], 'y': [1, np.nan], 'yy': [1, 2], 'z': [np.nan, np.nan]})
    print(df_temp)
    df_sum = df_temp.apply(sum_series_values_ommit_nan, keep_notalllist_notallnum_str=keep_notalllist_notallnum_str)
    print(df_sum)
    return


def test_boundary():
    # draft: df cancat, push same index-column items into a list and keep the first record grouped by share columns, ommit nan.
    # # 处理结果：
    # key_col + share_cols + union_cols
    # 其中：
    #     key_col唯一去重
    #     share_cols取按key_col聚集的group by的first
    #     union_cols取按key_col聚集的group by的全体元素列表
    df1 = pd.DataFrame({'key': [10], 'share': [12], 'a': [[1]], 'b': [np.nan]})
    df2 = pd.DataFrame({'key': [10], 'share': [12], 'a': [[11]], 'b': [[22]]})
    df3 = pd.DataFrame({'key': [20], 'share': [34], 'a': [[1, 11]], 'b': [np.nan]})
    df4 = pd.DataFrame({'key': [20], 'share': [34], 'a': [[1, 11]], 'b': [np.nan]})
    df5 = pd.DataFrame({'key': [30], 'share': [56], 'a': [[]], 'b': [[]]})
    df6 = pd.DataFrame({'key': [30], 'share': [56], 'a': [[]], 'b': [[]]})
    print(df1, df2, df3, df4, df5, df6)

    df_list = [df1, df2, df3, df4, df5, df6]  # 需输入

    df_raw_cancat = pd.concat(df_list)

    # df_raw_cancat.index.name = 'default_index' # Just for test

    print('\ndf_raw_cancat to be fromated:\n', df_raw_cancat)

    DEFAULT_INDEX_NAME = 'index'
    DF_INDEX_HAS_NAME = df_raw_cancat.index.name != None
    if not DF_INDEX_HAS_NAME:
        df_raw_cancat.index.name = DEFAULT_INDEX_NAME

    df_raw_cancat.reset_index(inplace=True)  # 将index列连同列名转为普通列

    print('\ndf_raw_cancat formated:\n', df_raw_cancat)

    print("===============")
    print('- df_raw_cancat.columns', list(df_raw_cancat.columns))
    on_col = 'key'  # 需输入
    if not on_col:
        print("union on_col is None, try to use df.index.name as default union on_col!")
    key_col = on_col or DEFAULT_INDEX_NAME
    print('- on_col =', on_col)
    gb = df_raw_cancat.groupby(by=key_col)

    union_cols = ['a', 'b']  # 需输入
    if key_col in union_cols:
        print("Error: union on_col = {key_col} should not be included in union_cols = {union_cols}!".format(
            key_col=key_col, union_cols=union_cols))
        # return
    print('- union_cols:', union_cols)
    share_cols = [c for c in df_raw_cancat.columns if c not in union_cols]  # 需输入
    # share_cols = ['share']  # 需输入
    if key_col not in share_cols:
        share_cols = [key_col] + share_cols
    print('- share_cols:', share_cols)
    df_union_groupby_dict = {}
    for idx, i_df in gb:
        print("-------")
        print(i_df[union_cols])
        df_union = pd.DataFrame(i_df[union_cols].apply(sum_series_values_ommit_nan, axis=0, keep_notalllist_notallnum_str='first'))
        print(df_union)
        df_union_groupby_dict[idx] = df_union
    print('================')
    df_union_cols = pd.concat(list(df_union_groupby_dict.values()))
    print(df_union_groupby_dict.keys())
    print(df_union_groupby_dict.values())
    df_union_cols[key_col] = list(df_union_groupby_dict.keys())

    print('\n1) df_union_cols:\n', df_union_cols)
    print('\n2.1) share_cols raw:\n', df_raw_cancat[share_cols])
    df_share_cols = df_raw_cancat[share_cols].groupby(by=key_col,
                                                      as_index=False).first()  # share_cols列，按key_col分组，每组取第一个
    print('\n2.2) share_cols filtered:\n', df_share_cols)
    assert (len(df_share_cols) == len(df_union_cols))
    merge_how__share_union = 'left'
    df_merge = pd.merge(df_share_cols, df_union_cols, on=key_col, how=merge_how__share_union)

    raw_index_name = df_raw_cancat.index.name
    if raw_index_name in share_cols:
        df_merge.set_index(raw_index_name, inplace=True)
    if not DF_INDEX_HAS_NAME:  # 恢复原始index命名
        df_merge.index.name = None
    print('\n3) df_merge:\n', df_merge)


if __name__ == '__main__':
    test_sum_nums_or_lists()
    test_sum_series_values_ommit_nan(keep_notalllist_notallnum_str='queue')
    test_boundary()
