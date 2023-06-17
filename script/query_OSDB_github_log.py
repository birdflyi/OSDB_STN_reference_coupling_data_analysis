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
pkg_rootdir = os.path.dirname(cur_dir)  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

import pandas as pd

from etc import filePathConf
from script import columns_simple
from utils.conndb import ConnDB


def query_repo_log_each_year_to_csv_dir(repo_names, columns, save_dir, sql_param=None, update_exist_data=False):
    conndb = ConnDB()
    columns_str = ', '.join(columns)
    sql_param = sql_param or {}
    sql_param = dict(sql_param)
    table = sql_param.get("table", "opensource.gh_events")
    start_end_year = sql_param.get("start_end_year", [2022, 2023])
    start_year = start_end_year[0]
    try:
        end_year = start_end_year[1]
    except IndexError:
        end_year = start_year + 1
    get_year_constraint = lambda x, y=None: f"created_at BETWEEN '{str(x)}-01-01 00:00:00' AND '{str(y or (x + 1))}-01-01 00:00:00'"

    # query and save
    for year in range(start_year, end_year):
        sql_ref_repo_pattern = f'''
        SELECT {{columns}} FROM {table} WHERE {get_year_constraint(year)} AND repo_name='{{repo_name}}';
        '''
        for repo_name in repo_names:
            repo_name_fileformat = repo_name.replace('/', '_')
            sql_ref_repo = sql_ref_repo_pattern.format(**{"columns": columns_str, "repo_name": repo_name})
            # print(sql_ref_repo)

            filename = f"{repo_name_fileformat}_{year}.csv"
            save_path = os.path.join(save_dir, filename)

            if update_exist_data or not os.path.exists(save_path):
                conndb.sql = sql_ref_repo
                conndb.execute()
                conndb.df_rs.to_csv(save_path)

                print(f"{filename} saved!")
            else:
                print(f"{filename} exists!")
    print("Done!")
    return


if __name__ == '__main__':
    # 1. 按repo_name分散存储到每一个csv文件中
    UPDATE_EXIST_DATA = False  # UPDATE SAVED RESULTS FLAG

    # 1.1 repo reference features as columns of sql
    columns = columns_simple

    # 1.2 get repo_names as condition of sql
    # repo_names = ['sqlite/sqlite', 'MariaDB/server', 'mongodb/mongo', 'redis/redis', 'elastic/elasticsearch', 'influxdata/influxdb', 'ClickHouse/ClickHouse', 'apache/hbase']
    df_OSDB_githubprj = pd.read_csv(os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                                 "dbfeatfusion_records_202306_automerged_manulabeled_with_repoid.csv"),
                                    header='infer', index_col=None)
    df_OSDB_githubprj = df_OSDB_githubprj[pd.notna(df_OSDB_githubprj["github_repo_id"])]  # filter github_repo_id must exist

    repo_names = list(df_OSDB_githubprj["github_repo_link"].values)

    # # test cases
    # tst_repo = ['sqlite/sqlite', 'MariaDB/server', 'mongodb/mongo', 'redis/redis', 'elastic/elasticsearch',
    #             'influxdata/influxdb', 'ClickHouse/ClickHouse', 'apache/hbase']
    # assert (all([r in repo_names for r in tst_repo]))
    # repo_names = tst_repo

    # 1.3 query and save
    save_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], "repos")
    sql_param = {
        "table": "opensource.gh_events",
        "start_end_year": [2022, 2023],
    }
    query_repo_log_each_year_to_csv_dir(repo_names, columns, save_dir, sql_param, update_exist_data=UPDATE_EXIST_DATA)
