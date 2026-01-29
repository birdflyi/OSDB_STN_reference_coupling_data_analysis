#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python 3.6

import os

__author__ = 'Lou Zehua <cs_zhlou@163.com>'
__time__ = '2019/6/16 0016 16:47'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Base directory: ROOT DIRECTORY

# ----------------------------------------------------------------------------------------------------------------------
#  Define file_names
# ----------------------------------------------------------------------------------------------------------------------
# Define the relative path of directory:
# Notes:  A directory must end with a suffix '_DIR' and end with '/' in the content.
#   A path ending with a suffix '_PATH' is recommended.
DATA_DIR = 0
GLOBAL_DATA_DIR = 1
GITHUB_OSDB_DATA_DIR = 2

DBMS_REPOS_RAW_CONTENT_DIR = 10
DBMS_REPOS_DEDUP_CONTENT_DIR = 11
DBMS_REPOS_GH_CORE_DIR = 12
DBMS_REPOS_GH_CORE_REF_NODE_AGG_DIR = 13

DBMS_REPOS_KEY_FEATS_PATH = 100

absPathDict = {
    DATA_DIR: os.path.join(BASE_DIR, 'data/'),
    GLOBAL_DATA_DIR: os.path.join(BASE_DIR, 'data/global_data/'),
    GITHUB_OSDB_DATA_DIR: os.path.join(BASE_DIR, 'data/github_osdb_data/'),
    DBMS_REPOS_RAW_CONTENT_DIR: os.path.join(BASE_DIR, 'data/github_osdb_data/repos'),
    DBMS_REPOS_DEDUP_CONTENT_DIR: os.path.join(BASE_DIR, 'data/github_osdb_data/repos_dedup_content'),
    DBMS_REPOS_GH_CORE_DIR: os.path.join(BASE_DIR, 'data/github_osdb_data/repos_GH_CoRE'),
    DBMS_REPOS_GH_CORE_REF_NODE_AGG_DIR: os.path.join(BASE_DIR, 'data/github_osdb_data/repos_GH_CoRE_ref_node_agg'),
    DBMS_REPOS_KEY_FEATS_PATH: os.path.join(
        BASE_DIR, "data/github_osdb_data/dbfeatfusion_records_202410_automerged_manulabeled_with_repoid.csv")
}

fileNameDict = {k: v.replace('\\', '/').split('/')[-1] for k, v in absPathDict.items()}

absDirDict = {k: '/'.join(v.replace('\\', '/').split('/')[:-1]) for k, v in absPathDict.items()}
