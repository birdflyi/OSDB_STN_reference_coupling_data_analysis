#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2026/1/20 19:24
# @Author : 'Lou Zehua'
# @File   : rep_gh_genser.py

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


import re


def replace_in_file(file_path, pattern, repl):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_data = file.read()

    flag_find_pattern_in_file_data = bool(re.compile(pattern).search(file_data))
    print(f"Find {pattern} in {file_path}: {flag_find_pattern_in_file_data}")

    if flag_find_pattern_in_file_data:
        file_data = re.sub(pattern, repl, file_data)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_data)


def replace_string_in_folder(folder_path, old_str, new_str):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            replace_in_file(file_path, old_str, new_str)


# 使用示例
folders = [
    'data/github_osdb_data/repos_GH_CoRE',
    ]
for folder_path in folders:
    old_str = 'GitHub_Other_Links'
    new_str = 'GitHub_GenSer_Other_Links'
    replace_string_in_folder(folder_path, old_str, new_str)
