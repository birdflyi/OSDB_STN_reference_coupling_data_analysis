#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/27 21:11
# @Author : 'Lou Zehua'
# @File   : query_repos_event_log.py
import logging
import os

import pandas as pd
from GH_CoRE import columns_simple
from GH_CoRE.utils.logUtils import setup_logging
from GH_CoRE.working_flow.query_OSDB_github_log import query_repo_log_each_year_to_csv_dir, get_repo_name_fileformat, \
    get_repo_year_filename

from etc import filePathConf, pkg_rootdir

setup_logging(base_dir=pkg_rootdir)
logger = logging.getLogger(__name__)


def query_OSDB_github_log_from_dbserver(key_feats_path=None, save_dir=None, update_exist_data=False, sql_param=None):
    # 1. 按repo_name分散存储到每一个csv文件中
    UPDATE_EXIST_DATA = update_exist_data  # UPDATE SAVED RESULTS FLAG
    # 1.1 repo reference features as columns of sql
    columns = columns_simple
    # 1.2 get repo_names as condition of sql
    # repo_names = ['sqlite/sqlite', 'MariaDB/server', 'mongodb/mongo', 'redis/redis', 'elastic/elasticsearch', 'influxdata/influxdb', 'ClickHouse/ClickHouse', 'apache/hbase']
    key_feats_path = key_feats_path or os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                                    "dbfeatfusion_records_202410_automerged_manulabeled_with_repoid.csv")
    df_OSDB_github_key_feats = pd.read_csv(key_feats_path, header='infer', index_col=None)
    df_OSDB_github_key_feats = df_OSDB_github_key_feats[
        pd.notna(df_OSDB_github_key_feats["github_repo_id"])]  # filter github_repo_id must exist
    repo_names = list(df_OSDB_github_key_feats["github_repo_link"].values)
    # 1.3 query and save
    save_dir = save_dir or os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR], "repos")
    sql_param = sql_param or {
        "table": "opensource.gh_events",
        "start_end_year": [2023, 2024],
    }
    query_repo_log_each_year_to_csv_dir(repo_names, columns, save_dir, sql_param, update_exist_data=UPDATE_EXIST_DATA)
    return repo_names


if __name__ == '__main__':
    year = 2023
    repo_names = ["TuGraph-family/tugraph-db", "facebook/rocksdb", "cockroachdb/cockroach"][0:2]
    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]

    if repo_names:
        sql_param = {
            "table": "opensource.events",
            "start_end_year": [year, year + 1],
        }
        query_repo_log_each_year_to_csv_dir(repo_names, columns=columns_simple, save_dir=dbms_repos_raw_content_dir,
                                            sql_param=sql_param)
    else:
        query_OSDB_github_log_from_dbserver(key_feats_path=dbms_repos_key_feats_path, save_dir=dbms_repos_raw_content_dir)

    filenames_exists = os.listdir(dbms_repos_raw_content_dir)
    if repo_names:
        repo_names_fileformat = list(map(get_repo_name_fileformat, repo_names))
        filenames = [get_repo_year_filename(s, year) for s in repo_names_fileformat]
        filenames = [filename for filename in filenames if filename in filenames_exists]
    else:
        filenames = filenames_exists
    logger.log(logging.INFO, msg=f"{len(filenames)} filenames: {filenames}")
