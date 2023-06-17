#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/1/30 19:17
# @Author : 'Lou Zehua'
# @File   : conndb.py

from __future__ import annotations

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

import time
import traceback

import pandas as pd

from sshtunnel import SSHTunnelForwarder
from clickhouse_driver import Client


class AuthConfig:
    # ----------------------------general final host---------------------------------
    SERVER_IP_LOCALHOST = 0
    SERVER_PORT_LOCALHOST = 1
    DBMS_USERNAME = 100
    DBMS_PASSWORD = 101
    USE_DATABASE = 102

    # ----------------------------local docker image---------------------------------
    # Local clickhouse GUI: http://ui.tabix.io/
    auth_settings_local_hosts = {
        SERVER_IP_LOCALHOST: "*",
        SERVER_PORT_LOCALHOST: 9000,

        DBMS_USERNAME: "*",
        DBMS_PASSWORD: "",
        USE_DATABASE: "*",
    }

    # ----------------------------------Aliyun---------------------------------------
    # Aliyun clickhouse GUI: https://signin.aliyun.com/1224904496484627.onaliyun.com/login.htm#/main
    SERVER_IP_INTMED_1 = -10
    SERVER_PORT_INTMED_1 = -11
    USERNAME_INTMED_1 = -12
    PASSWORD_INTMED_1 = -13
    SERVER_IP_TARGET = -20
    SERVER_PORT_TARGET = -21

    auth_settings_aliyun_hosts = {
        SERVER_IP_TARGET: "*",
        SERVER_PORT_TARGET: 3306,

        DBMS_USERNAME: "*",
        DBMS_PASSWORD: "*",
        USE_DATABASE: "*"
    }

    auth_settings_aliyun_intermediate_hosts = {
        SERVER_IP_INTMED_1: "*",
        SERVER_PORT_INTMED_1: 22,
        SERVER_IP_LOCALHOST: "*",
        SERVER_PORT_LOCALHOST: 10022,  # any available port.
        USERNAME_INTMED_1: "*",
        PASSWORD_INTMED_1: "*",  # replace '\' with '\\'.

        SERVER_IP_TARGET: "*",
        SERVER_PORT_TARGET: 3306,

        DBMS_USERNAME: "*",
        DBMS_PASSWORD: "*",
        USE_DATABASE: "*"
    }

    I_AUTH_SETTINGS_LOCAL_HOSTS = 0
    I_AUTH_SETTINGS_ALIYUN_HOSTS = 1
    I_AUTH_SETTINGS_ALIYUN_INTERMEDIATE_HOSTS = 2

    DEFAULT_INTMED_MODE = I_AUTH_SETTINGS_ALIYUN_HOSTS  # Just change your default AUTH_SETTINGS_HOSTS here :)

    auth_settings_dicts = {
        I_AUTH_SETTINGS_LOCAL_HOSTS: auth_settings_local_hosts,
        I_AUTH_SETTINGS_ALIYUN_HOSTS: auth_settings_aliyun_hosts,
        I_AUTH_SETTINGS_ALIYUN_INTERMEDIATE_HOSTS: auth_settings_aliyun_intermediate_hosts
    }
    default_auth_settings_dict = auth_settings_dicts[DEFAULT_INTMED_MODE]


class ConnDB:
    auth_settings_dict = AuthConfig.default_auth_settings_dict
    sql = None
    rs = None
    df_rs = None
    columns = None
    auto_update_columns = True
    client = None
    intmed_mode = AuthConfig.DEFAULT_INTMED_MODE

    def __init__(self, sql: str | None = None, intmed_mode: int | None = None, auto_update_columns: bool = True):
        self.sql = sql

        if intmed_mode is None:
            self.intmed_mode = ConnDB.intmed_mode
            self.auth_settings_dict = ConnDB.auth_settings_dict
        else:
            self.intmed_mode = intmed_mode
            self.auth_settings_dict = AuthConfig.auth_settings_dicts[intmed_mode]

        if self.intmed_mode == AuthConfig.I_AUTH_SETTINGS_ALIYUN_INTERMEDIATE_HOSTS:
            self.SERVER_IP_INTMED_1 = self.auth_settings_dict[AuthConfig.SERVER_IP_INTMED_1]
            self.SERVER_PORT_INTMED_1 = self.auth_settings_dict[AuthConfig.SERVER_PORT_INTMED_1]
            self.USERNAME_INTMED_1 = self.auth_settings_dict[AuthConfig.USERNAME_INTMED_1]
            self.PASSWORD_INTMED_1 = self.auth_settings_dict[AuthConfig.PASSWORD_INTMED_1]
        if self.intmed_mode in [AuthConfig.I_AUTH_SETTINGS_ALIYUN_HOSTS, AuthConfig.I_AUTH_SETTINGS_ALIYUN_INTERMEDIATE_HOSTS]:
            self.SERVER_IP_TARGET = self.auth_settings_dict.get(AuthConfig.SERVER_IP_TARGET)
            self.SERVER_PORT_TARGET = self.auth_settings_dict.get(AuthConfig.SERVER_PORT_TARGET)
        if self.intmed_mode in [AuthConfig.I_AUTH_SETTINGS_LOCAL_HOSTS, AuthConfig.I_AUTH_SETTINGS_ALIYUN_INTERMEDIATE_HOSTS]:
            self.SERVER_IP_LOCALHOST = self.auth_settings_dict.get(AuthConfig.SERVER_IP_LOCALHOST)
            self.SERVER_PORT_LOCALHOST = self.auth_settings_dict.get(AuthConfig.SERVER_PORT_LOCALHOST)

        self.DBMS_USERNAME = self.auth_settings_dict[AuthConfig.DBMS_USERNAME]
        self.DBMS_PASSWORD = self.auth_settings_dict[AuthConfig.DBMS_PASSWORD]
        self.USE_DATABASE = self.auth_settings_dict[AuthConfig.USE_DATABASE]

        self.auto_update_columns = auto_update_columns

    def query_clickhouse(self):
        try:
            if self.intmed_mode == AuthConfig.I_AUTH_SETTINGS_LOCAL_HOSTS:
                self.client = Client(host=self.SERVER_IP_LOCALHOST, port=self.SERVER_PORT_LOCALHOST,
                                     user=self.DBMS_USERNAME,
                                     password=self.DBMS_PASSWORD,
                                     database=self.USE_DATABASE,
                                     send_receive_timeout=600)
            elif self.intmed_mode == AuthConfig.I_AUTH_SETTINGS_ALIYUN_HOSTS:
                self.client = Client(host=self.SERVER_IP_TARGET, port=self.SERVER_PORT_TARGET,
                                     user=self.DBMS_USERNAME,
                                     password=self.DBMS_PASSWORD,
                                     database=self.USE_DATABASE,
                                     send_receive_timeout=600)
            elif self.intmed_mode == AuthConfig.I_AUTH_SETTINGS_ALIYUN_INTERMEDIATE_HOSTS:
                with SSHTunnelForwarder(
                        (self.SERVER_IP_INTMED_1, self.SERVER_PORT_INTMED_1),
                        ssh_username=self.USERNAME_INTMED_1,
                        ssh_password=self.PASSWORD_INTMED_1,
                        remote_bind_address=(self.SERVER_IP_TARGET, self.SERVER_PORT_TARGET),
                        local_bind_address=('0.0.0.0', self.SERVER_PORT_LOCALHOST)) as tunnel:
                    self.client = Client(host=self.SERVER_IP_LOCALHOST, port=tunnel.local_bind_port,
                                         user=self.DBMS_USERNAME,
                                         password=self.DBMS_PASSWORD,
                                         database=self.USE_DATABASE,
                                         send_receive_timeout=600)
            else:
                raise ValueError(f"The intmed_mode is expected in {list(AuthConfig.auth_settings_dicts.keys())}! "
                                 f"Got intmed_mode={self.intmed_mode}!")
            self.rs, column_types = self.client.execute(self.sql, with_column_types=True)
            temp_columns = [c for (c, t) in column_types]
            if self.auto_update_columns:
                self.columns = temp_columns
            else:
                self.columns = self.columns or temp_columns
        except BaseException as e:
            sql_log = self.sql[:500] + ("..." if self.sql[500:] else "")
            print("DB Exception happened while querying sql: \n\t{}\n".format(sql_log))
            print('Check the connection settings.\n' + traceback.format_exc())
            sys.exit()

        self.df_rs = pd.DataFrame(self.rs, columns=self.columns)
        return self.df_rs

    def execute(self, sql=None, columns=None, show_time_cost=False):
        self.sql = sql or self.sql
        self.auto_update_columns = not columns
        self.columns = columns or self.columns
        if not show_time_cost:
            self.query_clickhouse()
        else:
            start = time.time()
            self.query_clickhouse()
            end = time.time()
            print("Query time cost: {:.5f} s".format(end - start))
        return self.df_rs


if __name__ == '__main__':
    conndb = ConnDB(intmed_mode=AuthConfig.I_AUTH_SETTINGS_ALIYUN_HOSTS)
    default_table = 'opensource.gh_events'
    get_year_constraint = lambda x: f"created_at BETWEEN '{str(x)}-01-01 00:00:00' AND '{str(x + 1)}-01-01 00:00:00'"

    # conndb.sql = "SHOW databases;"

    columns = ["actor_id", "actor_login", "repo_id", "repo_name", "issue_id", "type", "action", "created_at", "pull_merged"]
    select_columns = ', '.join(columns)
    conndb.sql = f"SELECT {select_columns} FROM {default_table} where {get_year_constraint(2023)} LIMIT 10;"
    # show columns s.t. use table opensource.gh_events
    conndb.execute(show_time_cost=True)
    print("columns: ", conndb.columns)
    print("df_rs: \n", conndb.df_rs)

    # regenerate data_description.csv
    use_database = "opensource"
    use_table = "gh_events"
    # https://github.com/X-lab2017/open-digger/blob/master/docs/assets/data_description.csv
    conndb.sql = f"SELECT * FROM system.columns WHERE database='{use_database}' AND table='{use_table}';"
    df_data_description = conndb.execute(show_time_cost=True)
    data_description_path = os.path.join(pkg_rootdir, 'data/global_data/data_description.csv')
    df_data_description.to_csv(data_description_path, index=False, encoding='utf-8')
    print("columns: ", conndb.columns)
    print("df_data_description is saved!")
