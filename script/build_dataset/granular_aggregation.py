#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2026/1/17 17:41
# @Author : 'Lou Zehua'
# @File   : granular_aggregation.py

import json

import numpy as np
import pandas as pd
from GH_CoRE.model import Attribute_getter, ObjEntity
from GH_CoRE.working_flow.body_content_preprocessing import read_csvs

from etc import filePathConf


def granu_agg(row: pd.Series, repo_id=None):
    if row["src_entity_type"] == "Actor":
        row["src_entity_id_agg"] = row["src_entity_id"]
        row["src_entity_type_agg"] = row["src_entity_type"]
    else:
        row["src_entity_id_agg"] = "R_" + str(repo_id)
        row["src_entity_type_agg"] = "Repo"

    tar_entity_id_agg = None
    tar_entity_type_agg = "Object"
    tar_entity_objnt_prop_dict = parse_tar_entity_objnt_prop_dict(row["tar_entity_objnt_prop_dict"])
    if tar_entity_objnt_prop_dict:
        if "repo_id" in tar_entity_objnt_prop_dict.keys():
            if tar_entity_objnt_prop_dict["repo_id"] is not None:  # Except for unknown sha like fragment
                tar_entity_id_agg = "R_" + str(tar_entity_objnt_prop_dict["repo_id"])
                tar_entity_type_agg = "Repo"
        elif "actor_id" in tar_entity_objnt_prop_dict.keys():
            if tar_entity_objnt_prop_dict["actor_id"] is not None:
                tar_entity_id_agg = "A_" + str(tar_entity_objnt_prop_dict["actor_id"])
                tar_entity_type_agg = "Actor"
        else:
            pass  # can not parse
    row["tar_entity_id_agg"] = tar_entity_id_agg
    row["tar_entity_type_agg"] = tar_entity_type_agg
    return row


def set_entity_type_fine_grained(row: pd.Series):
    ent_type = "GitHub_Service_External_Links"
    tar_entity_objnt_prop_dict = parse_tar_entity_objnt_prop_dict(row["tar_entity_objnt_prop_dict"])
    need_check_objnt_prop = isinstance(tar_entity_objnt_prop_dict, dict) and ("repo_id" in tar_entity_objnt_prop_dict.keys() or "actor_id" in tar_entity_objnt_prop_dict.keys())
    if not need_check_objnt_prop:  # GitHub_Other_Service and GitHub_Service_External_Links and other wrong pattern has no id
        if row["tar_entity_match_pattern_type"] in ["GitHub_Other_Service", "GitHub_Service_External_Links"]:
            ent_type = row["tar_entity_match_pattern_type"]
        else:
            pass  # Can not get a valid node response from GitHub REST API or GitHub GraphQL. Regard as GitHub_Service_External_Links.
    else:  # row["tar_entity_type"] have Fine grained type when row["tar_entity_type"] != "Object", especially for Issue_PR and SHA pattern
        if row["tar_entity_type"] == "Object":
            ent_type = row["tar_entity_match_pattern_type"]
            if ent_type == "Issue_PR":
                if isinstance(tar_entity_objnt_prop_dict, dict):
                    repo_id = tar_entity_objnt_prop_dict.get("repo_id")
                    issue_number = tar_entity_objnt_prop_dict.get("issue_number")
                    if repo_id and issue_number:
                        row["tar_entity_type"] = Attribute_getter.__get_issue_type(repo_id, issue_number)
                        ent_type = row["tar_entity_type"]
                        tar_entity = ObjEntity(ent_type)
                        tar_entity.set_val(tar_entity_objnt_prop_dict)
                        row["tar_entity_id"] = tar_entity.__repr__(brief=True) if tar_entity.__PK__ else None
        else:
            ent_type = row["tar_entity_type"]  # for Issue, IssueComment, PullRequest, PullRequestReviewComment and Commit
    row["tar_entity_type_fine_grained"] = ent_type
    return row


def parse_tar_entity_objnt_prop_dict(tar_entity_objnt_prop_dict_raw):
    tar_entity_objnt_prop_dict = None
    try:
        if np.isnan(float(tar_entity_objnt_prop_dict_raw)):
            tar_entity_objnt_prop_dict = None
    except:
        pass

    if pd.isna(tar_entity_objnt_prop_dict_raw):  # all of GitHub_Other_Service, GitHub_Service_External_Links
        pass
    else:
        try:
            tar_entity_objnt_prop_dict = dict(tar_entity_objnt_prop_dict_raw)
        except Exception:
            prop_str = str(tar_entity_objnt_prop_dict_raw)
            try:
                tar_entity_objnt_prop_dict = json.loads(prop_str)
            except json.JSONDecodeError:
                # Swap the two quotation marks and try to parse again
                prop_str = prop_str.replace('"', '$').replace("'", '"').replace('$', "'")
                try:
                    # if prop_str.startswith("'") and prop_str.endswith("'"):
                    #     prop_str = prop_str[1:-1].replace("'", '"')
                    tar_entity_objnt_prop_dict = json.loads(prop_str)
                except json.JSONDecodeError:
                    try:
                        tar_entity_objnt_prop_dict = dict(eval(prop_str))
                    except Exception:
                        prop_str = prop_str.replace("'", '"')  # Forced analysis with [\', \"] mixed mode
                        tar_entity_objnt_prop_dict = json.loads(prop_str)
    return tar_entity_objnt_prop_dict


if __name__ == '__main__':
    year = 2023
    # dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    # dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]
    # dbms_repos_dedup_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_DEDUP_CONTENT_DIR]
    collaboration_relation_extraction_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_GH_CORE_DIR]
    repo_names = ["pingcap/tidb", "tikv/tikv"]
    filenames = [f"{name.replace('/', '_')}_{str(year)}" for name in repo_names]
    df_dbms_repos_dict = read_csvs(collaboration_relation_extraction_dir, filenames=filenames, index_col=None)
    df_dbms_repo = df_dbms_repos_dict[filenames[0]]
    # relation filter
    df_dbms_repo = df_dbms_repo[df_dbms_repo["relation_type"] == "Reference"]
    # target node granular aggregation
    df_dbms_repo = df_dbms_repo.apply(granu_agg)
    # G_repo = build_collab_net(df_dbms_repo, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
    #                           default_node_types=['src_entity_type', 'tar_entity_type'], default_edge_type="event_type",
    #                           init_record_as_edge_attrs=True, use_df_col_as_default_type=True, out_g_type='DG')
