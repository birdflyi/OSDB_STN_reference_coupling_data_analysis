#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/8 4:11
# @Author : 'Lou Zehua'
# @File   : authConf.py 

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


GITHUB_TOKENS = ['GITHUB_TOKEN_1', 'GITHUB_TOKEN_2']
