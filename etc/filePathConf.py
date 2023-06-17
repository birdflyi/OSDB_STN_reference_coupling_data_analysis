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

absPathDict = {
    DATA_DIR: os.path.join(BASE_DIR, 'data/'),
    GLOBAL_DATA_DIR: os.path.join(BASE_DIR, 'data/global_data/'),
    GITHUB_OSDB_DATA_DIR: os.path.join(BASE_DIR, 'data/github_osdb_data/'),
}

fileNameDict = {k: v.replace('\\', '/').split('/')[-1] for k, v in absPathDict.items()}

absDirDict = {k: '/'.join(v.replace('\\', '/').split('/')[:-1]) for k, v in absPathDict.items()}
