#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/4/20 17:33
# @Author : 'Lou Zehua'
# @File   : body_content_preprocessing.py 

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

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pylab import rcParams

rcParams['figure.figsize'] = 6, 4
rcParams['figure.dpi'] = 400
rcParams['savefig.dpi'] = 400

from etc import filePathConf
from script import columns_simple
from utils.conndb import ConnDB


def read_csvs(csv_dir, ignore_empty=True, header='infer', index_col=0, filenames=None, **kwargs):
    # read csvs in csv_dir.
    # e.g. If 'filename1.csv', 'filename2.csv' in csv_dir, return {'filename1': df1, 'filename2': df2}
    df_dict = {}
    filenames = filenames or os.listdir(csv_dir)
    for full_file_name in filenames:
        file_name, suffix = os.path.splitext(full_file_name)
        if suffix == '.csv':
            kwargs["low_memory"] = kwargs.get("low_memory", False)
            df = pd.read_csv(os.path.join(csv_dir, full_file_name), header=header, index_col=index_col, **kwargs)
            if ignore_empty and not len(df):
                print(file_name, 'is empty! It will be ignored.')
            else:
                df_dict[file_name] = df
    return df_dict


# SELECT distinct type FROM opensource.gh_events WHERE created_at between '2023-01-01 00:00:00' and '2023-05-01 00:00:00' and notEmpty(issue_title);
# RESULT1: [
#     'IssueCommentEvent',
#     'IssuesEvent',
#     'PullRequestEvent',
#     'PullRequestReviewCommentEvent',
#     'PullRequestReviewEvent',
# ]
# SELECT distinct type FROM opensource.gh_events WHERE created_at between '2023-01-01 00:00:00' and '2023-05-01 00:00:00' and notEmpty(body);
# RESULT2: [
#     'CommitCommentEvent',
#     'IssueCommentEvent',
#     'IssuesEvent',
#     'PullRequestEvent',
#     'PullRequestReviewEvent',
#     'PullRequestReviewCommentEvent',
# ]
# SELECT distinct type FROM opensource.gh_events WHERE created_at between '2023-01-01 00:00:00' and '2023-05-01 00:00:00' and notEmpty(push_commits.message);
# RESULT3: ['PushEvent']
# SELECT distinct type FROM opensource.gh_events WHERE created_at between '2023-01-01 00:00:00' and '2023-05-01 00:00:00' and notEmpty(release_body);
# RESULT4: ['ReleaseEvent']
def dedup_content(df_repo):  # 数据库存储时将repo_description等信息重复存储，需要过滤，保留首次遇到的值
    type_cols_dict = {
        'IssuesEvent': ['issue_title', 'body'],  # content: issue_title（action = closed reopened labeled冗余，可只取opened），body（action = closed reopened labeled冗余，可只取opened）
        'IssueCommentEvent': ['body'],  # content: body（非冗余）
        'PullRequestEvent': ['issue_title', 'body'],  # content: issue_title（action = closed reopened labeled冗余，可只取opened）, body（action = closed reopened labeled冗余，可只取opened）
        'PullRequestReviewEvent': ['body'],  # content: body（非冗余）
        'PullRequestReviewCommentEvent': ['body'],  # content: body（非冗余）
        'PushEvent': ['push_commits.message'],  # content: push_commits.message（非冗余）
        'CommitCommentEvent': ['body'],  # content: body（非冗余）
        'ReleaseEvent': ['release_body'],  # content: release_body（非冗余）
    }

    required_features = list(set(sum(list(type_cols_dict.values()), ['type', 'action'])))
    if set(list(df_repo.columns)) >= set(required_features):
        temp_df_repo = df_repo.copy()
        temp_df_repo.loc[temp_df_repo['action'].apply(lambda x: x in ['closed', 'reopened', 'labeled']), ['issue_title', 'body']] = np.nan  # for 'IssuesEvent' and 'PullRequestEvent'
        temp_df_repo.loc[temp_df_repo['type'].apply(lambda x: x in ['IssueCommentEvent', 'PullRequestReviewEvent', 'PullRequestReviewCommentEvent']), ['issue_title']] = np.nan  # for 'IssuesEvent' and 'PullRequestEvent'
    else:
        print('MissingColumnsError: The required_features:', required_features, '. Got columns:', list(df_repo.columns))
        return
    return temp_df_repo


if __name__ == '__main__':
    year = 2022
    # tst_repo = ['sqlite/sqlite', 'MariaDB/server', 'mongodb/mongo', 'redis/redis', 'elastic/elasticsearch',
    #             'influxdata/influxdb', 'ClickHouse/ClickHouse', 'apache/hbase']
    # filenames = [s.replace("/", "_") + f'_{year}.csv' for s in tst_repo]
    filenames = None

    # 读入csv，去除数据库存储时额外复制的重复issue信息
    dbms_repos_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], 'repos')
    df_dbms_repos_raw_dict = read_csvs(dbms_repos_dir, filenames=filenames)

    print(len(df_dbms_repos_raw_dict))

    DEDUP_CONTENT_OVERWRITE = False  # UPDATE SAVED RESULTS FLAG
    dbms_repos_dedup_content_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], 'repos_dedup_content')

    for repo_key, df_dbms_repo in df_dbms_repos_raw_dict.items():
        save_path = os.path.join(dbms_repos_dedup_content_dir, "{repo_key}.csv".format(**{"repo_key": repo_key}))
        if DEDUP_CONTENT_OVERWRITE or not os.path.exists(save_path):
            dedup_content(df_dbms_repo).to_csv(save_path)
    if not DEDUP_CONTENT_OVERWRITE:
        print('skip exist dedup_content...')
    print('dedup_content done!')
