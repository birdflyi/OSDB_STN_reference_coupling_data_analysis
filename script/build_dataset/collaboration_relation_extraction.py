#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/27 22:47
# @Author : 'Lou Zehua'
# @File   : collaboration_relation_extraction.py

import logging
import os
import traceback
from functools import partial

import pandas as pd
from GH_CoRE import columns_simple
from GH_CoRE.model.Relation_extraction import get_obj_collaboration_tuples_from_record, get_df_collaboration, \
    save_GitHub_Collaboration_Network
from GH_CoRE.utils import QueryCache
from GH_CoRE.utils.logUtils import setup_logging
from GH_CoRE.working_flow.body_content_preprocessing import read_csvs, dedup_content
from GH_CoRE.working_flow.query_OSDB_github_log import query_repo_log_each_year_to_csv_dir

from script import pkg_rootdir
from script.build_dataset.query_repos_event_log import query_OSDB_github_log_from_dbserver
from script.build_dataset.repo_filter import get_filenames_by_repo_names, get_intersection

setup_logging(base_dir=pkg_rootdir)
logger = logging.getLogger(__name__)

def process_body_content(raw_content_dir, processed_content_dir, filenames=None, dedup_content_overwrite_all=False):
    # reduce_redundancy
    # 读入csv，去除数据库存储时额外复制的重复issue信息
    df_dbms_repos_raw_dict = read_csvs(raw_content_dir, filenames=filenames, index_col=0)
    if not dedup_content_overwrite_all:
        print('Skip the exist dedup_content. Only process added files...')
    for repo_key, df_dbms_repo in df_dbms_repos_raw_dict.items():
        save_path = os.path.join(processed_content_dir, "{repo_key}.csv".format(**{"repo_key": repo_key}))
        if dedup_content_overwrite_all or not os.path.exists(save_path):
            dedup_content(df_dbms_repo).to_csv(save_path, header=True, index=True, encoding='utf-8', lineterminator='\n')
    print('dedup_content done!')
    return


def collaboration_relation_extraction(repo_keys, df_dbms_repos_dict, save_dir, repo_key_skip_to_loc=None,
                                      last_stop_index=None, limit=None, update_exists=True, add_mode_if_exists=True,
                                      cache_max_size=200, use_relation_type_list=None):
    """
    :param repo_keys: filenames right stripped by suffix `.csv`
    :param df_dbms_repos_dict: key: repo_keys, value: dataframe of dbms repos event logs
    :param save_dir: save the results for each `repo_key` in repo_keys into this directory
    :param repo_key_skip_to_loc: skip the indexes of repo keys smaller than repo_key_skip_to_loc in the order of df_dbms_repos_dict.keys()
    :param last_stop_index: set last_stop_index = -1 or None(by default) if skip nothing
    :param limit: set limit = -1 or None(by default) if no limit
    :param update_exists: only process repo_keys not exists old result when update_exists=False
    :param add_mode_if_exists: only takes effect when parameter update_exists=True
    :param cache_max_size: int type, set cache_max_size=-1 if you donot want to use any cache
    :param use_relation_type_list: to optionally extract the relation types in ['EventAction', 'Reference'], see event_trigger_ERE_triples_dict.
    :return: None
    """
    repo_key_skip_to_loc = repo_key_skip_to_loc if repo_key_skip_to_loc is not None else 0
    last_stop_index = last_stop_index if last_stop_index is not None else -1  # set last_stop_index = -1 if skip nothing
    # last_stop_index + 1,  this last_stop_index is the index of rows where the raw log file `id` column matches the
    # `event_id` column in the file result log file.
    rec_add_mode_skip_to_loc = last_stop_index + 1

    limit = limit if limit is not None else -1
    I_REPO_KEY = 0
    I_REPO_LOC = 1
    I_RECORD_LOC = 2
    process_checkpoint = ['', 0, 0]
    cache = QueryCache(max_size=cache_max_size) if cache_max_size > 0 else None
    cache.match_func = partial(QueryCache.d_match_func,
                               **{"feat_keys": ["link_pattern_type", "link_text", "rec_repo_id"]})
    try:
        for i, repo_key in enumerate(repo_keys):
            process_checkpoint[I_REPO_KEY] = repo_key
            process_checkpoint[I_REPO_LOC] = i
            if i < repo_key_skip_to_loc:
                continue
            df_repo = df_dbms_repos_dict[repo_key]
            save_path = os.path.join(save_dir, f'{repo_key}.csv')
            if os.path.exists(save_path) and not update_exists:
                continue

            for index, rec in df_repo.iterrows():
                process_checkpoint[I_RECORD_LOC] = index
                if limit > 0:
                    if index >= limit:
                        logger.info(
                            f"Processing progress: {repo_key}@{i}: [{rec_add_mode_skip_to_loc}: {index}]. Batch task completed!")
                        break
                if index < rec_add_mode_skip_to_loc:
                    continue
                obj_collaboration_tuple_list, cache = get_obj_collaboration_tuples_from_record(
                    rec, cache=cache, use_relation_type_list=use_relation_type_list)
                df_collaboration = get_df_collaboration(obj_collaboration_tuple_list, extend_field=True)
                save_GitHub_Collaboration_Network(df_collaboration, save_path=save_path, add_mode_if_exists=add_mode_if_exists)
            logger.info(f"Processing progress: {repo_key}@{i}#{process_checkpoint[I_RECORD_LOC]}: task completed!")
            rec_add_mode_skip_to_loc = 0
        logger.info(f"Processing progress: all task completed!")
    except BaseException as e:
        logger.info(
            f"Processing progress: {process_checkpoint[I_REPO_KEY]}@{process_checkpoint[I_REPO_LOC]}#{process_checkpoint[I_RECORD_LOC]}. "
            f"The process stopped due to an exception!")
        tb_lines = traceback.format_exception(e.__class__, e, e.__traceback__)
        logger.error(''.join(tb_lines))
    return


def collaboration_relation_extraction_service(dbms_repos_key_feats_path, dbms_repos_raw_content_dir,
                                              dbms_repos_dedup_content_dir, collaboration_relation_extraction_dir,
                                              repo_names=None, stop_repo_names=None, year=2023,
                                              validate_OSDB_github_repo_id=True):
    # Get github logs
    sql_param = {
        "table": "opensource.events",
        "start_end_year": [year, year + 1],
    }
    if repo_names is not None:
        query_repo_log_each_year_to_csv_dir(repo_names, columns=columns_simple, save_dir=dbms_repos_raw_content_dir,
                                            sql_param=sql_param)
    else:
        repo_names = query_OSDB_github_log_from_dbserver(key_feats_path=dbms_repos_key_feats_path,
                                                         save_dir=dbms_repos_raw_content_dir, sql_param=sql_param)
    if stop_repo_names:
        repo_names = [name for name in repo_names if name not in stop_repo_names]

    # Filter files
    filenames_exists = os.listdir(dbms_repos_raw_content_dir)
    if repo_names is not None:
        filenames_used = get_filenames_by_repo_names(repo_names, year)
    else:
        filenames_used = filenames_exists
    filenames_scope_list = [filenames_used, filenames_exists]
    filenames = get_intersection(filenames_scope_list)
    if validate_OSDB_github_repo_id:
        df_OSDB_github_key_feats = pd.read_csv(dbms_repos_key_feats_path, header='infer', index_col=None)
        df_OSDB_github_key_feats = df_OSDB_github_key_feats[
            pd.notna(df_OSDB_github_key_feats["github_repo_id"])]  # filter github_repo_id must exist
        repo_names_has_github_repo_id = list(df_OSDB_github_key_feats["github_repo_link"].values)
        filenames_has_github_repo_id = get_filenames_by_repo_names(repo_names_has_github_repo_id, year)
        filenames_scope_list = [filenames, filenames_has_github_repo_id]
        filenames = get_intersection(filenames_scope_list)
    logger.log(logging.INFO, msg=f"Use {len(filenames)} filenames: {filenames}")

    # Preprocess body content
    process_body_content(raw_content_dir=dbms_repos_raw_content_dir, processed_content_dir=dbms_repos_dedup_content_dir, filenames=filenames)
    df_dbms_repos_dict = read_csvs(dbms_repos_dedup_content_dir, filenames=filenames, index_col=0)

    # Get repo_keys
    d_repo_record_length = {k: len(df) for k, df in df_dbms_repos_dict.items()}
    d_repo_record_length_sorted = dict(sorted(d_repo_record_length.items(), key=lambda x: x[1], reverse=False))
    repo_keys = list(d_repo_record_length_sorted.keys())
    df_dbms_repos_dict = {k: df_dbms_repos_dict[k] for k in repo_keys}
    logger.log(logging.INFO, msg=f"Validated {len(repo_keys)} repo_keys sorted by the records count: {d_repo_record_length_sorted}")

    # Collaboration Relation extraction
    collaboration_relation_extraction(repo_keys, df_dbms_repos_dict, collaboration_relation_extraction_dir, update_exists=False,
                                      add_mode_if_exists=True, use_relation_type_list=["EventAction", "Reference"], last_stop_index=-1)


if __name__ == '__main__':
    from etc import filePathConf
    year = 2023
    dbms_repos_key_feats_path = filePathConf.absPathDict[filePathConf.DBMS_REPOS_KEY_FEATS_PATH]
    dbms_repos_raw_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_RAW_CONTENT_DIR]
    dbms_repos_dedup_content_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_DEDUP_CONTENT_DIR]
    collaboration_relation_extraction_dir = filePathConf.absPathDict[filePathConf.DBMS_REPOS_CORE_DIR]
    collaboration_relation_extraction_service(dbms_repos_key_feats_path, dbms_repos_raw_content_dir,
                                              dbms_repos_dedup_content_dir, collaboration_relation_extraction_dir,
                                              repo_names=None, stop_repo_names=None, year=year)
