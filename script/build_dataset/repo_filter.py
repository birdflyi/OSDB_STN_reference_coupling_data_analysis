#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/27 20:22
# @Author : 'Lou Zehua'
# @File   : repo_filter.py
import logging
import os

import pandas as pd
from GH_CoRE.utils.logUtils import setup_logging

from GH_CoRE.working_flow.query_OSDB_github_log import get_repo_name_fileformat, get_repo_year_filename

from etc import filePathConf, pkg_rootdir

setup_logging(base_dir=pkg_rootdir)
logger = logging.getLogger(__name__)


def get_intersection(lists):
    sets = tuple(map(set, lists))
    intersection = set.intersection(*sets)
    return list(intersection)


def get_filenames_by_repo_names(repo_names, year):
    if repo_names is not None:
        repo_names_fileformat = list(map(get_repo_name_fileformat, repo_names))
        filenames = [get_repo_year_filename(s, year) for s in repo_names_fileformat]
    else:
        filenames = None
    return filenames


if __name__ == '__main__':
    year = 2023
    repo_names = ["TuGraph-family/tugraph-db", "facebook/rocksdb", "cockroachdb/cockroach"][0:2]
    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]

    df_OSDB_github_key_feats = pd.read_csv(dbms_repos_key_feats_path, header='infer', index_col=None)
    df_OSDB_github_key_feats = df_OSDB_github_key_feats[
        pd.notna(df_OSDB_github_key_feats["github_repo_id"])]  # filter github_repo_id must exist
    repo_names_has_github_repo_id = list(df_OSDB_github_key_feats["github_repo_link"].values)
    filenames_has_github_repo_id = get_filenames_by_repo_names(repo_names_has_github_repo_id, year)
    filenames_exists = os.listdir(dbms_repos_raw_content_dir)
    if repo_names is not None:
        filenames_used = get_filenames_by_repo_names(repo_names, year)
    else:
        filenames_used = filenames_exists

    filenames_scope_list = [filenames_used, filenames_exists, filenames_has_github_repo_id]
    filenames = get_intersection(filenames_scope_list)

    logger.log(logging.INFO, msg=f"Use {len(filenames)} filenames: {filenames}")
