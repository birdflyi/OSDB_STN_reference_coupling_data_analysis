#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time     : 2021/2/4 20:19
# @Author   : 'Lou Zehua'
# @File     : profiles.py
# To avoid naming conflict, change the filename from 'profile' to 'profiles':
#   https://stackoverflow.com/questions/57323834/attributeerror-module-profile-has-no-attribute-run

encoding = 'utf-8'

# ----------------------------------------------------------------------------------------------------------------------
#  Default settings
# ----------------------------------------------------------------------------------------------------------------------
DATABASES = ["_temporary_and_external_tables", "default", "github_log", "system"]
TABLES = ['crx_period_activity', 'events', 'year2015', 'year2016', 'year2017', 'year2018', 'year2019', 'year2020', 'year2021', 'year2022']

# ----------------------------------------------------------------------------------------------------------------------
#  Authentication settings
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------general final host---------------------------------
SERVER_IP = 0
SERVER_PORT = 1
USERNAME = 2
PASSWORD = 3
MYSQL_USERNAME = 10
MYSQL_PASSWORD = 11
USE_DATABASE = 12

# ----------------------------local docker image---------------------------------
# Local clickhouse GUI: http://ui.tabix.io/
auth_settings_local_hosts = {
    SERVER_IP: "*",
    SERVER_PORT: 9000,
    MYSQL_USERNAME: "default",
    MYSQL_PASSWORD: "",
    USE_DATABASE: "*",
}

# ----------------------------------Aliyun---------------------------------------
# Aliyun clickhouse GUI: https://signin.aliyun.com/1224904496484627.onaliyun.com/login.htm#/main
SERVER_IP_INTMED_1 = -10
SERVER_PORT_INTMED_1 = -11
USERNAME_INTMED_1 = -12
PASSWORD_INTMED_1 = -13
SERVER_IP_INTMED_2 = -20
SERVER_PORT_INTMED_2 = -21
USERNAME_INTMED_2 = -22
PASSWORD_INTMED_2 = -23

auth_settings_aliyun_intermediate_hosts = {
    SERVER_IP_INTMED_1: "*",
    SERVER_PORT_INTMED_1: 22,
    USERNAME_INTMED_1: "*",
    PASSWORD_INTMED_1: "*",  # replace '\' with '\\'.

    SERVER_IP_INTMED_2: "*",
    SERVER_PORT_INTMED_2: 3306,
    USERNAME_INTMED_2: None,
    PASSWORD_INTMED_2: None,

    SERVER_IP: '127.0.0.1',
    SERVER_PORT: 10022,
    MYSQL_USERNAME: "*",
    MYSQL_PASSWORD: "*",
    USE_DATABASE: "*"
}

auth_settings_dicts = [auth_settings_local_hosts, auth_settings_aliyun_intermediate_hosts]
INDEX_A_S_LOCAL_HOSTS = 0  # local hosts has a snapshot of Aliyun clickhouse database, which can lead to a fast but low recall result.
INDEX_A_S_ALIYUN_INTERMEDIATE_HOSTS = 1
# intermediate pattern: 0(for local clickhouse) or 1(for query Aliyun clickhouse)
DEFAULT_INTMED_PAT = INDEX_A_S_ALIYUN_INTERMEDIATE_HOSTS
default_auth_settings_dict = auth_settings_dicts[DEFAULT_INTMED_PAT]
