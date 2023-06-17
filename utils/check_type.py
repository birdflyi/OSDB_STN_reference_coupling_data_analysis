#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/5/23 1:38
# @Author : 'Lou Zehua'
# @File   : check_type.py 

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


def is_number(s, nan_as_true=False):
    try:
        float_s = float(s)
        if nan_as_true:
            return True
        elif float_s == float_s:  # filter nan
            return True
        else:
            return False
    except TypeError or ValueError:
        return False


def is_list(x, nan_as_true=False, use_eval=False):
    flag = type(x) is list
    if nan_as_true:
        flag = flag or is_nan(x)
    if use_eval and type(x) is str:
        flag = str(x).startswith('[') and str(x).endswith(']')
    return flag


def is_str(x, nan_as_true=False):
    flag = type(x) is str
    if nan_as_true:
        flag = flag or is_nan(x)
    return flag


def is_nan(s):
    s = str(s)
    if s.lower() == 'nan':
        return True
    else:
        try:
            float_s = float(s)
        except ValueError:
            return False
        return float_s != float_s


def is_na(s, check_str_eval=True):
    if is_nan(s):
        return True
    try:
        if check_str_eval and type(s) == str:
            if len(s):
                s = eval(s)  # True: [np.nan, '', [], (), {}, np.array([]), pd.DataFrame(), '[]', '()', '{}']
        if not len(s):
            return True
    except ValueError:
        pass
    return False


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    empty_list = [np.nan, 'NaN', 'nan', '', [], (), {}, np.array([]), pd.DataFrame(), '[]', '()', '{}']
    empty_series = pd.Series(data=empty_list)
    print("data:")
    print(pd.DataFrame(data=[empty_series.apply(str), empty_series.apply(type)]).T)
    print("is_nan:")
    print(empty_series.apply(is_nan))
    print("is_na:")
    print(empty_series.apply(is_na))
