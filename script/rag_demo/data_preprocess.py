#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time   : 2026/2/18 17:57
# @Author : 'Lou Zehua'
# @File   : data_preprocess.py

import os
import sys

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = os.path.dirname(os.path.dirname(cur_dir))  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

import GH_CoRE
import json
import pandas as pd

from etc import filePathConf
from typing import Dict, Any

# # --- 1. 使用你提供的最新映射表 ---
# ENTITY_METADATA = {
#     "Actor": {"ABBR": "A"},
#     "Branch": {"ABBR": "B"},
#     "Commit": {"ABBR": "C"},
#     "CommitComment": {"ABBR": "CC"},
#     "Gollum": {"ABBR": "G"},
#     "Issue": {"ABBR": "I"},
#     "IssueComment": {"ABBR": "IC"},
#     "PullRequest": {"ABBR": "PR"},
#     "PullRequestReview": {"ABBR": "PRR"},
#     "PullRequestReviewComment": {"ABBR": "PRRC"},
#     "Push": {"ABBR": "P"},
#     "Release": {"ABBR": "RE"},
#     "Repo": {"ABBR": "R"},
#     "Tag": {"ABBR": "T"},
#     "Object": {"ABBR": "OBJ"},
#     "None": {"ABBR": "NONE"},
# }
#
# # 构建反向映射
# abbr_to_full_map = {attr['ABBR']: entity_name for entity_name, attr in ENTITY_METADATA.items()}

# --- 1. 动态构建映射表 ---
# 参考源码: https://github.com/birdflyi/GitHub_Collaboration_Relation_Extraction/blob/main/GH_CoRE/model/Entity_model.py
# 格式: {实体类名: {'ABBR': '前缀', ...}, ...}
# 构建反向映射: { "A": "Actor", "CC": "CommitComment", ... }
abbr_to_full_map = {v.get('ABBR'): k for k, v in GH_CoRE.model.Entity_model.ObjEntity.E.items() if v.get('ABBR')}


def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """安全解析 JSON 字符串"""
    if not json_str or not isinstance(json_str, str) or json_str.strip() in ['', 'nan']:
        return {}
    try:
        if json_str.startswith("'") and json_str.endswith("'"):
            json_str = json_str[1:-1].replace("'", '"')
        else:
            json_str = json_str.replace("'", '"')
        return json.loads(json_str)
    except Exception as e:
        return {}


def get_repo_repr(repo_id, repo_name) -> str:
    """
    从 repo_id (通常是数字) 中提取仓库名。
    注意：在 GH_CoRE 中，repo_id 通常是数据库 ID (数字)。
    如果 JSON 中没有提供 friendly name，这里只能用 ID。
    """
    # 这里可以扩展为查询数据库或缓存，目前简单返回 ID 或占位符
    repo_type_prefix = "Repo:"
    if repo_id or repo_name:
        repo_id_repr = f"[ID:{str(repo_id)}]" if repo_id else ""
        repo_name_repr = str(repo_name) if repo_name else ""
        repo_repr = repo_type_prefix + repo_id_repr + repo_name_repr
    else:
        repo_repr = repo_type_prefix + "Unknown"
    return repo_repr


def get_actor_repr(actor_id, actor_login) -> str:
    # 这里可以扩展为查询数据库或缓存，目前简单返回 ID 或占位符
    actor_type_prefix = "Actor:"
    if actor_id or actor_login:
        actor_id_repr = f"[ID:{str(actor_id)}]" if actor_id else ""
        actor_login_repr = f"@{actor_login}" if actor_login else ""
        actor_repr = actor_type_prefix + actor_id_repr + actor_login_repr
    else:
        actor_repr = actor_type_prefix + "Unknown"
    return actor_repr


def parse_id_string(entity_id: str, entity_type: str, prop_dict: Dict[str, Any] = {}) -> str:
    """
    核心函数：根据实体类型和 ID 字符串生成可读文本。
    现在会尝试从 ID 或 JSON 中提取 repo_id 以确保全局唯一。
    """

    if not entity_id or entity_id == entity_type:
        return f"{entity_type}:Unknown"

    # 优先尝试从 JSON 中获取 repo_id 和友好名称
    json_repo_id = prop_dict.get('repo_id', None)
    json_repo_name = prop_dict.get('repo_name', None)  # 如果有全名
    json_actor_id = prop_dict.get('actor_id', None)  # Actor
    json_actor_login = prop_dict.get('actor_login', None)  # Actor
    json_branch = prop_dict.get('branch_name', None)
    json_tag = prop_dict.get('tag_name', None)

    # --- 辅助函数：获取带仓库前缀的标识 ---
    def make_qualified_name(basic_name: str) -> str:
        """构造带仓库上下文的名称"""
        repo_repr = get_repo_repr(json_repo_id, json_repo_name)
        if json_repo_name or json_repo_id:
            return f"{repo_repr}:{basic_name}"
        else:
            # 如果 ID 字符串中包含 repo_id (通常在开头)，尝试提取
            # 简单逻辑：取 ID 的第一部分 (按非字母数字分割)
            potential_repo_part = entity_id.split(':')[0].split('#')[0].split('.')[0].split('-')[0]
            if potential_repo_part.isdigit() and len(potential_repo_part) > 0:
                repo_repr = get_repo_repr(potential_repo_part, json_repo_name)
                return f"{repo_repr}:{basic_name}"
            else:
                return basic_name

    # --- 1. Actor: 纯 ID ---
    if entity_type == "Actor":
        actor_id = entity_id.lstrip('A_')
        actor_repr = get_actor_repr(actor_id, json_actor_login)
        return actor_repr

    # --- 2. Branch: {repo_id}:{branch_name} ---
    elif entity_type == "Branch" and ':' in entity_id:
        try:
            # 格式: repo_id:branch_name
            repo_id, branch_name = entity_id.split(':', 1)
            branch_name = branch_name.split('#')[0]  # 清理
            # 优先使用 JSON 中的 branch_name，否则用 ID 中的
            name = json_branch or branch_name
            return make_qualified_name(f"Branch:{name}")
        except:
            pass

    # --- 3. Commit: {repo_id}@{commit_sha} ---
    elif entity_type == "Commit" and '@' in entity_id:
        try:
            # 注意：Commit 的唯一性主要靠 SHA
            repo_id, commit_sha = entity_id.split('@', 1)
            commit_sha = commit_sha.split('#')[0]
            short_sha = commit_sha[:7]
            # 仍然加上 Repo 上下文以便溯源
            sha_name = f"Commit[{short_sha}]"
            # 如果有 repo_name 用 repo_name，否则用 repo_id
            repo_repr = get_repo_repr(repo_id, json_repo_name)
            return f"{repo_repr}:{sha_name}"
        except:
            pass

    # --- 4. CommitComment: 格式 "{repo_id}@{commit_sha}#r{comment_id}" ---
    elif entity_type == "CommitComment" and '@' in entity_id:
        try:
            # 先按 #r 分割，获取前面的 repo@sha 部分
            repo_sha_part, comment_id = entity_id.split('#r', 1)
            repo_id, commit_sha = repo_sha_part.split('@')
            short_sha = commit_sha[:7]
            # 仍然加上 Repo 上下文以便溯源
            sha_name = f"CommentOnCommit[{short_sha}]"
            # 如果有 repo_name 用 repo_name，否则用 repo_id
            repo_repr = get_repo_repr(repo_id, json_repo_name)
            return f"{repo_repr}:{sha_name}:CommitComment[ID:{comment_id}]"
        except:
            pass

    # --- 5. Gollum: {repo_id}:wiki ---
    elif entity_type == "Gollum" and ':' in entity_id:
        try:
            repo_id, page_info = entity_id.split(':', 1)
            page_name = "WikiPage" if page_info == "wiki" else f"Wiki:{page_info}"
            return make_qualified_name(page_name)
        except:
            pass

    # --- 6. Issue / PullRequest: {repo_id}#{issue_number} ---
    elif entity_type in ["Issue", "PullRequest"] and '#' in entity_id:
        try:
            parts = entity_id.split('#')
            repo_id = parts[0]
            issue_number = None
            for part in parts[1:]:
                if part.isdigit():
                    issue_number = part
                    break
            if issue_number:
                base_name = f"{entity_type}#{issue_number}"
                repo_repr = get_repo_repr(repo_id, json_repo_name)
                return f"{repo_repr}:{base_name}"
        except:
            pass

    # --- 7. IssueComment: {repo_id}#{issue_number}#{comment_id} ---
    elif entity_type == "IssueComment" and entity_id.count('#') >= 2:
        try:
            # 格式: repo_id#issue_number#comment_id
            parts = entity_id.split('#')
            repo_id = parts[0]
            issue_number = parts[1]
            comment_id = parts[2]
            # 构造基础名
            base_name = f"CommentOnIssue#{issue_number}:IssueComment[ID:{comment_id}]"
            # 手动注入 repo_id 上下文
            repo_repr = get_repo_repr(repo_id, json_repo_name)
            return f"{repo_repr}:{base_name}"
        except:
            pass

    # --- 8. PullRequestReview (PRR): 格式 "{repo_id}#{issue_number}#prr-{review_id}" ---
    elif entity_type == "PullRequestReview":
        try:
            if '#prr-' in entity_id:
                repo_id_issue_number, review_id = entity_id.split('#prr-')
                repo_id, issue_number = repo_id_issue_number.split('#')
            base_name = f"PRReview#{review_id}"
            repo_repr = get_repo_repr(repo_id, json_repo_name)
            return f"{repo_repr}:ReviewOnPullRequest#{issue_number}:{base_name}"
        except:
            pass

    # --- 9. PullRequestReviewComment: {repo_id}#{issue_number}#r{comment_id} ---
    elif entity_type == "PullRequestReviewComment" and '#r' in entity_id:
        try:
            # 格式: repo_id#issue_number#rcomment_id
            base_part, comment_id = entity_id.split('#r', 1)
            repo_id, issue_number = base_part.split('#')[0], base_part.split('#')[-1]
            base_name = f"PRReviewComment#{comment_id}"
            repo_repr = get_repo_repr(repo_id, json_repo_name)
            return f"{repo_repr}:CommentOnPullRequest#{issue_number}:{base_name}"
        except:
            pass

    # --- 10. Push: {repo_id}.{push_id} ---
    elif entity_type == "Push" and '.' in entity_id:
        try:
            repo_id, push_id = entity_id.split('.', 1)
            base_name = f"PushEvent#{push_id}"
            repo_repr = get_repo_repr(repo_id, json_repo_name)
            return f"{repo_repr}:{base_name}"
        except:
            pass

    # --- 11. Release: {repo_id}-{release_id} ---
    elif entity_type == "Release" and '-' in entity_id:
        try:
            repo_id, release_id = entity_id.split('-', 1)
            name = json_tag or release_id
            base_name = f"Release:{name}"
            repo_repr = get_repo_repr(repo_id, json_repo_name)
            return f"{repo_repr}:{base_name}"
        except:
            pass

    # --- 12. Repo: 纯 ID ---
    elif entity_type == "Repo":
        # 如果 JSON 中有名字，显示名字，否则显示 ID
        repo_id = entity_id
        repo_repr = get_repo_repr(repo_id, json_repo_name)
        return repo_repr

    # --- 13. Tag: {repo_id}-{tag_name} ---
    elif entity_type == "Tag" and '-' in entity_id:
        try:
            repo_id, tag_name = entity_id.split('-', 1)
            name = json_tag or tag_name
            base_name = f"Tag:{name}"
            repo_repr = get_repo_repr(repo_id, json_repo_name)
            return f"{repo_repr}:{base_name}"
        except:
            pass

    # --- 默认回退 ---
    # 即使是默认回退，也尝试加上 Repo 上下文
    try:
        # 尝试从 ID 开头提取 repo_id
        potential_repo = entity_id.split(':')[0].split('#')[0].split('.')[0].split('-')[0]
        if potential_repo.isdigit() and len(potential_repo) > 3:  # 粗略判断是 ID
            repo_repr = get_repo_repr(potential_repo, json_repo_name)
            return f"{repo_repr}:{entity_type}[{entity_id[-10:]}]"
    except:
        pass
    return f"{entity_type}:{entity_id[:15]}..."


def get_entity_name(entity_id: str, parse_tar: bool = True, objnt_prop_dict: str = None, match_text: str = None,
                    tar_entity_type_fine_grained: str = None, src_default_repo_prop=None) -> str:
    """主解析函数"""
    empty_entity_id = not entity_id or not isinstance(entity_id, str) or entity_id.strip() in ['', 'nan']

    if not parse_tar:
        objnt_prop_dict = None
        match_text = None
        tar_entity_type_fine_grained = None
        if empty_entity_id:
            return "Null"

    elif empty_entity_id:
        if match_text and isinstance(match_text, str):
            if tar_entity_type_fine_grained:
                return f"Ref {tar_entity_type_fine_grained}:{match_text}"
            else:
                return f"Ref Unknown:{match_text}"
        return "Null"

    entity_id = entity_id.strip().strip('"')
    prefix = None
    id_body = entity_id
    if "_" in entity_id:
        prefix, id_body = entity_id.split("_", 1)
    entity_type = abbr_to_full_map.get(prefix, "Unknown")

    # 解析 JSON 属性
    if parse_tar:
        prop_dict = safe_json_loads(objnt_prop_dict)
    else:
        prop_dict = src_default_repo_prop if src_default_repo_prop is not None else {}

    # 优先解析 ID 字符串 (传入 prop_dict 以获取 repo_name 等上下文)
    parsed_name = parse_id_string(id_body, entity_type, prop_dict)

    if parse_tar:
        # 兜底逻辑 (如果 JSON 中有更友好的 Actor 名称)
        if entity_type == "Actor":
            if 'actor_id' in prop_dict and 'actor_login' in prop_dict:
                if str(prop_dict['actor_id']) in parsed_name:  # 确认prop_dict中的actor_id一致
                    return get_actor_repr(prop_dict['actor_id'], prop_dict['actor_login'])
        if 'actor_login' in prop_dict and '@' not in parsed_name:
            parsed_name += f"@{prop_dict['actor_login']}"
            return parsed_name
        if 'at_str' in prop_dict and '@' not in parsed_name:
            parsed_name += f"@{prop_dict['at_str']}"
            return parsed_name
        if entity_type == "Repo":
            if 'repo_id' in prop_dict.keys() and 'repo_name' in prop_dict.keys():
                if str(prop_dict['repo_id']) in parsed_name:  # 确认prop_dict中的repo_id一致
                    return get_repo_repr(prop_dict['repo_id'], prop_dict['repo_name'])

    return parsed_name


def parse_entity_string(tar_entity_name: str) -> dict:
    """
    解析 tar_entity_name 字符串，提取根实体及其属性，以及最终的实体类型。

    参数:
        tar_entity_name (str): 符合描述格式的字符串，例如：
            "Repo:[ID:123]owner/name"
            "Repo[ID:123]owner/name#123"
            "Actor:@actor_login"
            "Repo:owner/name:Commit[SHA:a1b2c3d]:CommitComment[ID:1234]"
            "Issue:123"
            "Issue#456"
            "Repo[ID:123]owner/name:Issue#456"

    返回:
        dict: 包含以下字段的字典（解析失败或不存在时值为 None）：
            - root_entity_type: 根实体类型（只有 Repo 或 Actor 才可能有值）
            - entity_type: 最终实体类型
            - repo_id: 仓库ID (int)
            - repo_name: 仓库名称
            - actor_id: 操作者ID (int)
            - actor_login: 操作者登录名
    """
    # 初始化结果字典，所有字段默认为 None
    result = {
        'root_entity_type': None,
        'entity_type': None,
        'repo_id': None,
        'repo_name': None,
        'actor_id': None,
        'actor_login': None
    }

    if not tar_entity_name:
        return result  # 输入为空，返回全 None

    # 辅助函数：从字符串中提取实体类型（忽略 # 及其后的内容）
    def extract_entity_type(s: str) -> str:
        # 先找到第一个 # 的位置
        hash_pos = s.find('#')
        if hash_pos != -1:
            s = s[:hash_pos]

        # 提取方括号前的类型名
        bracket_pos = s.find('[')
        if bracket_pos != -1:
            return s[:bracket_pos]
        return s

    # 1. 检查是否以 Repo: 或 Actor: 开头（标准格式）
    if tar_entity_name.startswith(('Repo:', 'Actor:')):
        # 标准格式处理
        first_colon = tar_entity_name.find(':')
        root_type = tar_entity_name[:first_colon]
        result['root_entity_type'] = root_type

        # 获取剩余部分
        remaining = tar_entity_name[first_colon + 1:]

        # 查找下一个不在方括号内的冒号
        next_colon = -1
        depth = 0
        for i, ch in enumerate(remaining):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
            elif ch == ':' and depth == 0:
                next_colon = i
                break

        if next_colon == -1:
            # 没有更多冒号，整个剩余部分都是属性
            prop_str = remaining
            remaining = ''
        else:
            # 有更多段，属性到第一个冒号前
            prop_str = remaining[:next_colon]
            remaining = remaining[next_colon + 1:]

        # 解析属性字符串
        if prop_str:
            # 检查是否有 ID
            if prop_str.startswith('[ID:') and ']' in prop_str:
                id_end = prop_str.find(']')
                id_part = prop_str[4:id_end]  # 跳过 '[ID:'
                if id_part.isdigit():
                    id_value = int(id_part)
                    if root_type == 'Repo':
                        result['repo_id'] = id_value
                    else:  # Actor
                        result['actor_id'] = id_value

                # 剩余部分作为名称（直到遇到 # 为止）
                name_part = prop_str[id_end + 1:].strip()
                # 如果名称部分包含 #，只取 # 之前的部分
                hash_pos = name_part.find('#')
                if hash_pos != -1:
                    name_part = name_part[:hash_pos]

                if name_part:
                    if root_type == 'Repo':
                        result['repo_name'] = name_part
                    else:  # Actor
                        if name_part.startswith('@'):
                            name_part = name_part[1:]
                        result['actor_login'] = name_part
            else:
                # 无 ID，整个字符串就是名称
                name_part = prop_str.strip()
                # 如果名称部分包含 #，只取 # 之前的部分
                hash_pos = name_part.find('#')
                if hash_pos != -1:
                    name_part = name_part[:hash_pos]

                if name_part:
                    if root_type == 'Repo':
                        result['repo_name'] = name_part
                    else:  # Actor
                        if name_part.startswith('@'):
                            name_part = name_part[1:]
                        result['actor_login'] = name_part

        # 解析最终的 entity_type
        if remaining:
            # 如果剩余部分包含 #，只取 # 之前的部分
            hash_pos = remaining.find('#')
            if hash_pos != -1:
                remaining = remaining[:hash_pos]

            # 查找最后一个不在方括号内的冒号
            last_colon = -1
            depth = 0
            for i, ch in enumerate(remaining):
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                elif ch == ':' and depth == 0:
                    last_colon = i

            if last_colon != -1:
                last_seg = remaining[last_colon + 1:]
            else:
                last_seg = remaining

            result['entity_type'] = extract_entity_type(last_seg)
        else:
            result['entity_type'] = root_type

    # 2. 检查是否以 Repo[ 或 Actor[ 开头（紧凑格式）
    elif tar_entity_name.startswith(('Repo[', 'Actor[')):
        # 找到根实体类型结束的位置
        bracket_pos = tar_entity_name.find('[')
        root_type = tar_entity_name[:bracket_pos]
        result['root_entity_type'] = root_type

        # 找到匹配的 ']'
        remaining = tar_entity_name[bracket_pos:]
        bracket_end = -1
        depth = 0
        for i, ch in enumerate(remaining):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    bracket_end = i
                    break

        if bracket_end != -1:
            # 提取ID部分
            id_part = remaining[1:bracket_end]  # 去除开头的 '['
            if id_part.startswith('ID:'):
                id_value = id_part[3:]
                if id_value.isdigit():
                    id_value = int(id_value)
                    if root_type == 'Repo':
                        result['repo_id'] = id_value
                    else:  # Actor
                        result['actor_id'] = id_value

            # 获取名称部分（从 ']' 后面开始，直到遇到下一个 ':' 或 '#')
            remaining = remaining[bracket_end + 1:]

            # 查找下一个 ':' 或 '#'
            next_sep = -1
            for i, ch in enumerate(remaining):
                if ch in (':', '#'):
                    next_sep = i
                    break

            if next_sep != -1:
                name_part = remaining[:next_sep].strip()
                remaining = remaining[next_sep:]
            else:
                name_part = remaining.strip()
                remaining = ''

            # 如果名称部分包含 #，只取 # 之前的部分
            hash_pos = name_part.find('#')
            if hash_pos != -1:
                name_part = name_part[:hash_pos]

            if name_part:
                if root_type == 'Repo':
                    result['repo_name'] = name_part
                else:  # Actor
                    if name_part.startswith('@'):
                        name_part = name_part[1:]
                    result['actor_login'] = name_part

            # 解析最终的 entity_type
            if remaining:
                # 如果剩余部分以 ':' 开头，去掉它
                if remaining.startswith(':'):
                    remaining = remaining[1:]

                # 如果剩余部分包含 #，只取 # 之前的部分
                hash_pos = remaining.find('#')
                if hash_pos != -1:
                    remaining = remaining[:hash_pos]

                if remaining:
                    # 查找最后一个不在方括号内的冒号
                    last_colon = -1
                    depth = 0
                    for i, ch in enumerate(remaining):
                        if ch == '[':
                            depth += 1
                        elif ch == ']':
                            depth -= 1
                        elif ch == ':' and depth == 0:
                            last_colon = i

                    if last_colon != -1:
                        last_seg = remaining[last_colon + 1:]
                    else:
                        last_seg = remaining

                    result['entity_type'] = extract_entity_type(last_seg)
                else:
                    result['entity_type'] = root_type
            else:
                result['entity_type'] = root_type
        else:
            result['entity_type'] = root_type

    else:
        # 3. 其他格式（如 Issue#456, Issue:123 等）
        # 直接解析最终实体类型，忽略 # 及其后的所有内容
        result['entity_type'] = extract_entity_type(tar_entity_name)

    return result

def is_reponame_repokey_matched(repo_name: str, repo_key: str, year=2023):
    from GH_CoRE.working_flow import get_repo_name_fileformat, get_repo_year_filename
        
    match_flag = False
    repo_name_fileformat = get_repo_name_fileformat(repo_name)
    filename = get_repo_year_filename(repo_name_fileformat, year)
    if filename == repo_key + '.csv':
        match_flag = True
    return match_flag
    

def get_repo_id_by_repo_key(repo_key, df_repo_i_pr_rec_cnt, year=2023):
    repo_id_match_flags = df_repo_i_pr_rec_cnt.apply(
        lambda row: str(row['repo_id']) if is_reponame_repokey_matched(row['repo_name'], repo_key, year) else None, axis=1)
    repo_id = repo_id_match_flags.dropna().iloc[0] if not repo_id_match_flags.dropna().empty else None
    return repo_id
    

def load_and_transform_csv(csv_path: str, year=2023, only_ref_relation=True):
    from langchain.schema import Document
    from script.build_dataset.granular_aggregation import set_entity_type_fine_grained
    
    repo_key = os.path.splitext(os.path.basename(csv_path))[0]
    github_osdb_data_dir = filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR]
    repo_i_pr_rec_cnt_path = os.path.join(github_osdb_data_dir, 'repo_activity_statistics/repo_i_pr_rec_cnt.csv')
    df_repo_i_pr_rec_cnt = pd.read_csv(repo_i_pr_rec_cnt_path, index_col=None)
    df_repo_i_pr_rec_cnt = df_repo_i_pr_rec_cnt.fillna("")
    repo_id = get_repo_id_by_repo_key(repo_key, df_repo_i_pr_rec_cnt)
    repo_name = repo_key.rstrip(f"_{year}").replace("_", "/", 1)

    df = pd.read_csv(csv_path, index_col=None, encoding='utf-8', dtype=str)
    df = df.apply(set_entity_type_fine_grained, axis=1)
    if only_ref_relation:
        df = df[df["relation_type"] == "Reference"]
    documents = []
    df = df.fillna('')

    src_default_repo_prop = {
        "repo_id": repo_id,
        "repo_name": repo_name,
    }

    for _, row in df.iterrows():
        src_entity_id = row['src_entity_id']
        src_entity_name = get_entity_name(
            src_entity_id,
            False,
            src_default_repo_prop=src_default_repo_prop
        )

        tar_entity_id = row['tar_entity_id']
        tar_entity_name = get_entity_name(
            tar_entity_id,
            True,
            row.get('tar_entity_objnt_prop_dict', ''),
            row.get('tar_entity_match_text', ''),
            row.get("tar_entity_type_fine_grained", '')
        )
        tar_meta = parse_entity_string(tar_entity_name)

        relation_type = row['relation_type']
        relation = row['relation_label_repr']
        content = f"{src_entity_name} --({relation_type}::{relation})--> {tar_entity_name}"

        if row['event_trigger']:
            content += f" [EventTrigger: {row['event_trigger']}]"

        metadata = {
            "src_entity_id": src_entity_id,
            "src_entity_type": row["src_entity_type"],
            "src_repo_id": src_default_repo_prop["repo_id"],
            "src_repo_name": src_default_repo_prop["repo_name"],
            "tar_entity_id": tar_entity_id,
            "tar_entity_type": tar_meta.get("entity_type"),
            "tar_repo_id": tar_meta.get("repo_id"),
            "tar_repo_name": tar_meta.get("repo_name"),
            "event_id": row['event_id'],
            "event_type": row['event_type'],
            "timestamp": row['event_time'],
        }
        if tar_meta.get("root_entity_type") == "Actor":
            metadata["tar_actor_id"] = tar_meta.get("actor_id")
            metadata["tar_actor_login"] = tar_meta.get("actor_login")
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    return documents


if __name__ == '__main__':
    demo_path = os.path.abspath(
        os.path.join(filePathConf.absPathDict[filePathConf.DBMS_REPOS_GH_CORE_DIR], 'apache_kylin_2023.csv'))
    docs = load_and_transform_csv(demo_path)
    # with open("./documents.json", "w", encoding="utf-8") as file:
    #     json.dump([doc.dict() for doc in docs], file, ensure_ascii=False, indent=4)
    for doc in docs:
        if doc.metadata["src_entity_id"] == "PR_28738447#2113":
            print(str(doc))
