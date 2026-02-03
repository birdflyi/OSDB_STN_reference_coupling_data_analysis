#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/3/29 17:34
# @Author : 'Lou Zehua'
# @File   : validate.py
import itertools
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

import re
import pandas as pd

from collections import Iterable
from datetime import datetime

from GH_CoRE.model import Attribute_getter
from GH_CoRE.utils.request_api import RequestGitHubAPI


class ValidateFunc:

    def __init__(self):
        pass

    @staticmethod
    def _str_has_substr_in_check_list(s, check_list, both_preprocess=None):
        if pd.isna(s):
            return False
        s = str(s)
        if both_preprocess is not None:
            s = both_preprocess(s)
        flag = False
        for e in check_list:
            if both_preprocess is not None:
                e = both_preprocess(e)
            osl_pattern = e + "\s*[vV]?\d*"
            if re.findall(osl_pattern, s):
                flag = True  # set flag=True if any pattern matched
        return flag

    @staticmethod
    def _has_common_opensource_license(s, match_abbr=True, strict_mode=15, single_elem_series_squeeze_out=True):
        """
        :param s: licenses content str.
        :param match_abbr: match a full name like "GNU General Public License" and its abbr "GPL" if True else only match a full name.
        :param strict_level:
            - OSI_CERT = 1(0000 0001) Only OSI certification,
            - BEYOND_SOFTWARE = 2(0000 0010) Open source without dispute, but not limited to software,
            - RESTRICT_REDISTRIBUTION = 4(0000 0100) Include controversial open-source licenses: "Server Side Public License" restricts redistribution to emphasize the obligation of open source
            - NOT_PROTECTED = 8(0000 1000) Include not protected: "Public Domain" lack of formal authorization framework for agreements
            - RESTRICT_USAGE = 16(0001 0000) Include licenses source-available: "Business Source License"
        :param single_elem_series_squeeze_out: set True when s is a licenses content with pandas.Series type.
        :return: a bool value or a bool value series with the same length as s.
        """
        OSI_CERT = 1
        BEYOND_SOFTWARE = 2
        RESTRICT_REDISTRIBUTION = 4
        NOT_PROTECTED = 8
        RESTRICT_USAGE = 16
        common_osl_fullname = []
        common_osl_abbr = []

        if strict_mode & OSI_CERT:
            # common open source licenses:
            # 1. OSI(Open Source Initiative) report from https://opensource.org/proliferation-report/
            # - Licenses that are popular and widely used or with strong communities (9):
            common_osl_fullname += [
                "Apache License, 2.0",  # Apache v2, Apache 2.0, Apache License
                "Apache License 2.0",  # Apache 2.0, Apache v2.0, Apache v2, Apache License, Apache
                "Apache License",  # Apache
                "3-Clause BSD License",  # BSD-3-Clause
                "New BSD license",  # BSD-3-Clause
                "Modified BSD License",  # BSD-3-Clause
                "Berkeley Software Distribution",  # BSD
                "GNU General Public License (GPL version 2)",  # GPL v2
                "GNU General Public License v3.0",  # GNU GPLv3, GPL v3
                "GNU Lesser General Public License v3.0",  # GNU LGPLv3, LGPL v3
                "GNU General Public License",  # GPL
                "GNU Library or “Lesser” General Public License (LGPL version 2)",  # LGPL v2
                "GNU Library General Public License",  # LGPL
                "GNU Lesser General Public License",  # LGPL
                "The MIT license",  # MIT
                "MIT license",  # MIT
                "Mozilla Public License 1.1 (MPL)",  # MPL 1.1
                "Mozilla Public License 2.0",  # MPL 2.0, MPL v2.0
                "Mozilla Public License",  # MPL
                "Common Development and Distribution License",  # CDDL
                "Common Public License",  # CPL
                "Eclipse Public License",  # EPL
            ]
            common_osl_abbr += ["Apache", "BSD", "GPL", "LGPL", "MIT", "MPL", "CDDL", "CPL", "EPL"]

            # 2. International licenses from https://opensource.org/licenses-old/category
            common_osl_fullname += [
                "CeCILL License 2.1",  # CECILL-2.1
                "CeCILL License",  # CECILL
                "European Union Public License",  # EUPL-1.2
                "Licence Libre du Québec – Permissive",  # LiLiQ-P
                "Licence Libre du Québec – Réciprocité",  # LiLiQ-R
                "Licence Libre du Québec – Réciprocité forte",  # LiLiQ-R+, LiLiQ-Rplus
                "Licence Libre du Québec – Réciprocité forte version 1.1",  # LiLiQ-Rplus-1.1
                "Mulan Permissive Software License，Version 2",  # Mulan PSL v2, MulanPSL-2.0
                "Mulan Permissive Software License v2",  # MulanPSL-2.0, MulanPSL – 2.0
                "Mulan Permissive Software License",  # Mulan PSL, MulanPSL
                "Mulan Public License，Version 2",  # Mulan PubL v2,
                "Mulan Public License",  # Mulan PubL, Mulan
                "MulanOWL",
            ]
            common_osl_abbr += ["CeCILL", "EUPL", "LiLiQ-P", "LiLiQ-R", "LiLiQ-R+", "LiLiQ-Rplus", "Mulan"]

            # 3. Licenses that are redundant with more popular licenses, Non-reusable licenses, Superseded licenses
            #   from https://opensource.org/licenses-old/category
            common_osl_fullname += [
                "OpenLDAP Public License",  # OLDAP
                "The PostgreSQL Licence",  # PostgreSQL
                "PostgreSQL Licence",  # PostgreSQL
                "Zope Public License",  # ZPL

                "Apple Public Source License",  # APSL
                "Python License",  # Python

                "Open Software License",  # OSL
            ]
            common_osl_abbr += ["OLDAP", "PostgreSQL", "ZPL", "APSL", "Python", "OSL"]

            # 4. Uncategorized Licenses from https://opensource.org/licenses-old/category
            common_osl_fullname += [
                "Boost Software License 1.0",  # BSL 1.0, BSL v1.0, BSL-1.0
                "Boost Software License",  # Boost
                "Cryptographic Autonomy License",  # CAL
                "Common Public Attribution License",   # CPAL
                "GNU Affero General Public License version 3",  # AGPL-3.0
                "GNU Affero General Public License v3.0",  # GNU AGPLv3, AGPL v3
                "GNU Affero General Public License",  # AGPL
                "ISC License",  # ISC
                "Microsoft Public License",  # MS-PL
            ]
            common_osl_abbr += ["Boost", "CAL", "CPAL", "AGPL", "ISC", "MS-PL"]

        if strict_mode & BEYOND_SOFTWARE:
            # 5. other open source license
            common_osl_fullname += [
                # from https://creativecommons.org/licenses/by/4.0/
                "Creative Commons License",  # CC
                "Creative Commons",  # CC
            ]
            common_osl_abbr += ["CC"]

        if strict_mode & RESTRICT_REDISTRIBUTION:
            # 6. Warning: controversial open-source license!!!
            common_osl_fullname += [
                "Eclipse Distribution License",  # EDL
                "Parity Public License",
                "Server Side Public License",  # https://opensource.org/blog/the-sspl-is-not-an-open-source-license
            ]
            common_osl_abbr += ["EDL", "SSPL"]

        if strict_mode & NOT_PROTECTED:
            # 7. Warning: controversial open-source license!!!
            common_osl_fullname += [
                "Public Domain",  # https://opensource.org/blog/public-domain-is-not-open-source
            ]
            # common_osl_abbr += []

        if strict_mode & RESTRICT_USAGE:
            # 8. Warning: controversial open-source license!!!
            common_osl_fullname += [
                "Business Source License",  ## Source code is guaranteed to become Open Source at a certain point in time.
                "LLAMA 2 COMMUNITY LICENSE",  # LLAMA  ## https://opensource.org/blog/metas-llama-license-is-still-not-open-source
                "Source Available",  # https://opensource.org/press-mentions/source-available-is-not-open-source-and-thats-okay-dries-buytaert
            ]
            common_osl_abbr += ["LLAMA"]

        common_osl_fullname = list(set(common_osl_fullname))
        common_osl_abbr = list(set(common_osl_abbr))
        check_list = common_osl_fullname + common_osl_abbr if match_abbr else common_osl_fullname

        both_preprocess = lambda s: str(s).lower()
        if isinstance(s, Iterable):
            s = pd.Series(s)
            flag = s.apply(ValidateFunc._str_has_substr_in_check_list, check_list=check_list, both_preprocess=both_preprocess)
            if single_elem_series_squeeze_out and len(flag) == 1:
                flag = flag[0]
        else:
            s = str(s)
            flag = ValidateFunc._str_has_substr_in_check_list(s, check_list=check_list, both_preprocess=both_preprocess)
        return flag

    @staticmethod
    def check_open_source_license(series, nan_as_final_false=False, only_common_osl=False):
        series = pd.Series(series)
        if nan_as_final_false:
            # state: NaN ×, "Y/N" √, "Y" √, "N" ×
            flag = True if "Y" in str(series["open_source_license"]) else False
        else:  # keep nan license info to Wait for API query to complete values
            # state: NaN √, "Y/N" √, "Y" √, "N" ×
            flag = True if str(series["open_source_license"]) != "N" else False
        if only_common_osl:
            flag = flag and ValidateFunc._has_common_opensource_license(series["License_info"])
        return flag

    @staticmethod
    def has_open_source_github_repo_link(series):
        flag = False
        series = pd.Series(series)
        if pd.notna(series["github_repo_link"]):
            flag = True if series["github_repo_link"] != '-' else False
        return flag

    @staticmethod
    def has_open_source_github_repo_id(series):
        flag = False
        series = pd.Series(series)
        if pd.notna(series["github_repo_id"]):
            flag = True if series["github_repo_id"] != '-' else False
        return flag


def get_license_name_by_repo_id(repo_id):
    license_name = None
    requestGitHubAPI = RequestGitHubAPI(url_pat_mode="id")
    url = requestGitHubAPI.get_url("repo", params={"repo_id": repo_id})
    response = requestGitHubAPI.request(url)
    if response is not None and hasattr(response, "json"):
        data = response.json()
        license_dict = data.get("license", None)
        if isinstance(license_dict, dict):
            license_name = license_dict["name"]
    else:
        license_name = '-'
        print(f"repo_id: {repo_id}. Empty data.")
    return license_name


def complete_license_info(row_ser, update_license_by_API=False):
    row_ser = pd.Series(row_ser)
    old_value = row_ser.get("License_info", None)
    if not pd.isna(old_value) and not update_license_by_API:
        license_name = old_value
    else:
        github_repo_link = row_ser["github_repo_link"]
        if pd.isna(github_repo_link):
            license_name = None if pd.isna(old_value) else old_value
        elif github_repo_link == "-":
            license_name = "-" if pd.isna(old_value) else old_value
        elif '/' in github_repo_link:
            repo_id = row_ser.get("github_repo_id", None)
            if pd.isna(repo_id):
                repo_id = Attribute_getter.__get_repo_id_by_repo_full_name(github_repo_link)
            license_name = get_license_name_by_repo_id(repo_id)
        else:
            raise ValueError("github_repo_link should be separated by '/' or set to a value in ['-', None]!")
    return license_name


def complete_github_repo_id(row_ser, update_repo_id_by_API=False):
    row_ser = pd.Series(row_ser)
    old_value = row_ser.get("github_repo_id", None)
    if not pd.isna(old_value) and not update_repo_id_by_API:
        repo_id = old_value
    else:
        github_repo_link = row_ser["github_repo_link"]
        if pd.isna(github_repo_link):
            repo_id = None
        elif github_repo_link == "-":
            repo_id = "-"
        elif '/' in github_repo_link:
            repo_id_from_github_api = Attribute_getter.__get_repo_id_by_repo_full_name(github_repo_link)
            repo_id = repo_id_from_github_api if pd.notna(repo_id_from_github_api) else '-'  # api查询不到即不存在
        else:
            raise ValueError("github_repo_link should be separated by '/' or set to a value in ['-', None]!")
    return repo_id


def get_repo_created_at_by_repo_id(repo_id, auto_parse=False):
    repo_created_at = None
    requestGitHubAPI = RequestGitHubAPI(url_pat_mode="id")
    url = requestGitHubAPI.get_url("repo", params={"repo_id": repo_id})
    response = requestGitHubAPI.request(url)
    if response is not None and hasattr(response, "json"):
        data = response.json()
        created_at_str = data.get("created_at", None)
        if isinstance(created_at_str, str):
            repo_created_at = datetime.strptime(created_at_str, '%Y-%m-%dT%H:%M:%SZ') if auto_parse else created_at_str
    else:
        repo_created_at = '-'
        print(f"repo_id: {repo_id}. Empty data.")
    return repo_created_at


def complete_repo_created_at(row_ser, update_repo_created_at_by_API=False):
    row_ser = pd.Series(row_ser)
    old_value = row_ser.get("repo_created_at", None)
    if not pd.isna(old_value) and not update_repo_created_at_by_API:
        repo_created_at = old_value
    else:
        github_repo_link = row_ser["github_repo_link"]
        if pd.isna(github_repo_link):
            repo_created_at = None if pd.isna(old_value) else old_value
        elif github_repo_link == "-":
            repo_created_at = "-" if pd.isna(old_value) else old_value
        elif '/' in github_repo_link:
            repo_id = row_ser.get("github_repo_id", None)
            if pd.isna(repo_id):
                repo_id = Attribute_getter.__get_repo_id_by_repo_full_name(github_repo_link)
            repo_created_at = get_repo_created_at_by_repo_id(repo_id, auto_parse=False)
        else:
            raise ValueError("github_repo_link should be separated by '/' or set to a value in ['-', None]!")
    return repo_created_at


if __name__ == '__main__':
    from etc import filePathConf

    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    df_OSDB_github_key_feats = pd.read_csv(dbms_repos_key_feats_path, header='infer', index_col=None, dtype=str)
    # ser_validate_osl = df_OSDB_github_key_feats.apply(ValidateFunc.check_open_source_license, only_common_osl=False, axis=1)
    # print(df_OSDB_github_key_feats[~ser_validate_osl])
    ser_validate_common_osl = df_OSDB_github_key_feats.apply(ValidateFunc.check_open_source_license, only_common_osl=True, axis=1)
    print(df_OSDB_github_key_feats[~ser_validate_common_osl])

    df_OSDB_github_key_feats["License_info"] = df_OSDB_github_key_feats.apply(complete_license_info, update_license_by_API=False, axis=1)
    ser_license_updated_validate_common_osl = df_OSDB_github_key_feats.apply(ValidateFunc.check_open_source_license, nan_as_final_false=True, only_common_osl=True, axis=1)
    print(df_OSDB_github_key_feats[~ser_license_updated_validate_common_osl])
    df_OSDB_github_key_feats.to_csv(dbms_repos_key_feats_path, header=True, index=False)
