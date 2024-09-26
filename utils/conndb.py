#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/1/30 19:17
# @Author : 'Lou Zehua'
# @File   : conndb.py

from __future__ import annotations

import sys
import time
import traceback

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
    # Download a ClickHouse sample data for your docker container:
    #   https://github.com/X-lab2017/open-digger/blob/master/sample_data/README.md#current-sample-datasets
    # Local clickhouse GUI: http://ui.tabix.io/
    auth_settings_local_hosts = {
        SERVER_IP_LOCALHOST: "<SERVER_IP_LOCALHOST>",
        SERVER_PORT_LOCALHOST: 9000,

        DBMS_USERNAME: "default",
        DBMS_PASSWORD: "",
        USE_DATABASE: 'default',
    }

    # ----------------------------------Aliyun---------------------------------------
    # Aliyun clickhouse GUI: https://signin.aliyun.com/login.htm#/main
    SERVER_IP_INTMED_1 = -10
    SERVER_PORT_INTMED_1 = -11
    USERNAME_INTMED_1 = -12
    PASSWORD_INTMED_1 = -13
    SERVER_IP_TARGET = -20
    SERVER_PORT_TARGET = -21

    auth_settings_aliyun_hosts = {
        SERVER_IP_TARGET: "<SERVER_IP>",
        SERVER_PORT_TARGET: 9000,  # TCP Port: 9000 or 3306, see https://help.aliyun.com/zh/clickhouse/support/faq

        DBMS_USERNAME: '<USERNAME_DBMS>',
        DBMS_PASSWORD: '<PASSWORD_DBMS>',
        USE_DATABASE: 'default'
    }   # updated in the file https://xlab2017.yuque.com/me1x4f/vti721/eabel7

    auth_settings_aliyun_intermediate_hosts = {
        SERVER_IP_INTMED_1: "<SERVER_IP_INTMED>",
        SERVER_PORT_INTMED_1: 22,
        SERVER_IP_LOCALHOST: '127.0.0.1',
        SERVER_PORT_LOCALHOST: 10022,  # any available port.
        USERNAME_INTMED_1: "<USERNAME_SERVER_INTMED>",
        PASSWORD_INTMED_1: "<PASSWORD_SERVER_INTMED>",  # replace '\' with '\\'.

        SERVER_IP_TARGET: "<SERVER_IP>",
        SERVER_PORT_TARGET: 3306,

        DBMS_USERNAME: '<USERNAME_DBMS>',
        DBMS_PASSWORD: '<PASSWORD_DBMS>',
        USE_DATABASE: 'default'
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
    df_format = True
    rs = None
    columns = None
    auto_update_columns = True
    client = None
    show_time_cost = False
    intmed_mode = AuthConfig.DEFAULT_INTMED_MODE

    def __init__(self, sql: str | None = None, intmed_mode: int | None = None, auto_update_columns: bool | None = None):
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

        if auto_update_columns is not None:
            self.auto_update_columns = auto_update_columns

    def query(self, *args, **kwargs):
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
            self.rs = self.client.query_dataframe(*args, **kwargs)
            if self.auto_update_columns:
                self.columns = self.rs.columns
            else:
                self.columns = self.columns or self.rs.columns
        except BaseException as e:
            sql_log = self.sql[:500] + ("..." if self.sql[500:] else "")
            print("DB Exception happened while querying sql: \n\t{}\n".format(sql_log))
            print('Check the connection settings.\n' + traceback.format_exc())
            sys.exit()
        if not self.df_format:
            self.rs = self.rs.apply(lambda x: tuple(x), axis=1).values.tolist()
        return self.rs

    def execute(self, sql=None, columns=None, df_format: bool | None = None, show_time_cost: bool | None = None):
        self.sql = sql or self.sql
        self.auto_update_columns = not columns
        self.columns = columns or self.columns
        if df_format is not None:
            self.df_format = df_format
        if show_time_cost is not None:
            self.show_time_cost = show_time_cost

        if not show_time_cost:
            self.query(self.sql, replace_nonwords=False)
        else:
            start = time.time()
            self.query(self.sql, replace_nonwords=False)
            end = time.time()
            print("Query time cost: {:.5f} s".format(end - start))
        return self.rs


if __name__ == '__main__':
    conndb = ConnDB(intmed_mode=AuthConfig.I_AUTH_SETTINGS_ALIYUN_HOSTS)
    default_table = 'opensource.events'
    get_year_constraint = lambda x: f"created_at BETWEEN '{str(x)}-01-01 00:00:00' AND '{str(x + 1)}-01-01 00:00:00'"

    # conndb.sql = "SHOW databases;"

    # conndb.sql = f'select count(*) AS cnt FROM {default_table} where {get_year_constraint(2023)};'

    columns = ["actor_id", "actor_login", "repo_id", "repo_name", "issue_id", "type", "action", "created_at", "pull_merged"]
    select_columns = ', '.join(columns)
    conndb.sql = f"SELECT {select_columns} FROM {default_table} where {get_year_constraint(2023)} LIMIT 10;"

    conndb.df_format = True
    conndb.execute(show_time_cost=True)
    print("columns: ", conndb.columns)
    print("rs: \n", conndb.rs)

    import os

    from utils import pkg_rootdir

    # regenerate data_description.csv
    use_database = "opensource"
    use_table = "events"
    # https://github.com/X-lab2017/open-digger/blob/master/docs/assets/data_description.csv
    conndb.sql = f"SELECT * FROM system.columns WHERE database='{use_database}' AND table='{use_table}';"
    df_data_description = conndb.execute(show_time_cost=True)
    data_description_path = os.path.join(pkg_rootdir, 'data/global_data/data_description.csv')
    df_data_description.to_csv(data_description_path, index=False, encoding='utf-8')
    print("columns: ", conndb.columns)
    print("df_data_description is saved!")
