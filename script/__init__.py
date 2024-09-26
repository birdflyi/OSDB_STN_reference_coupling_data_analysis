#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2023/4/19 21:16
# @Author : 'Lou Zehua'
# @File   : __init__.py.py 

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



'''markdown
# 1. 使用交叉引用网络的背景
    # Scio-Technical网络不仅仅是协作关系，从reference角度构建网络，可用来研究引用影响力reference influence
    # 1.1 参考文献
        # https://www.sciencedirect.com/science/article/abs/pii/S0950584919300527
        # Blincoe, K., Harrison, F., Kaur, N., & Damian, D. (2019). Reference coupling: An exploration of inter-project technical dependencies and their characteristics within large software ecosystems. Information and Software Technology, 110, 174-189.
        
# 2 数据集
    # 2.1 数据库领域仓库筛选范围
        # Aliyun DMS: https://dms.aliyun.com/?dbType=ClickHouse&instanceId=cc-uf6s6ckq946aiv4jy&instanceSource=RDS&regionId=cn-shanghai
        # sql：SELECT * FROM year2021  Where type='{type_value}'  and repo_name in ['sqlite/sqlite', 'MariaDB/server', 'mongodb/mongo', 
        # 'redis/redis', 'elastic/elasticsearch', 'influxdata/influxdb', 'ClickHouse/ClickHouse', 'apache/hbase'] LIMIT 10;
        # 相关变更：https://xlab2017.yuque.com/me1x4f/vti721/otpgpb8w1psromqz
    # 2.2 数据收集流程标准化
        # 2.2.1 种子数据
            # 从DB_Engines列表中topN的数据库中找出开源的数据库（DB_Engines是第三方机构，榜单中的数据库列表与科研机构CMU的dbdb.io中的数据库列表有不少重合，与国内第三方评估机构墨天轮榜单中的数据库列表有较大不同）
            # 筛选与引用网络有关的属性
            # 从clickhouse中查询数据库相关的信息，并存储到本地
        # 2.2.2 扩展数据
            # 2.2.2.a 从种子数据的属性列中筛选出数据项中含有交叉引用的特征
            # 2.2.2.b 识别其中的交叉引用并初步分析其分布
            # 2.2.2.c 仅研究同一领域项目中含引用量最高的项目的特性
            # 2.2.2.d 根据引用的仓库完成一阶同质扩展，根据引用的人员和issue、文件等信息完成一阶异质扩展

'''

'''
# Event feature

'IssuesEvent': ['issue_title', 'body'],  # content: issue_title（action = closed reopened labeled冗余，可只取opened）, body（action = closed reopened labeled冗余，可只取opened）; related_columns: id, type, action, actor_id, actor_login, *repo_id*, repo_name, org_id, org_login, created_at, `issue_id`, issue_number, **issue_title**, **body**, [issue_labels.name, issue_labels.color, issue_labels.default, issue_labels.description, *issue_author_id*, *issue_author_login*, issue_author_type, issue_author_association, *issue_assignee_id*, *issue_assignee_login*, *issue_assignees.login*, *issue_assignees.id*, issue_created_at, issue_updated_at, issue_comments, issue_closed_at]

'IssueCommentEvent': ['body'],  # content: ~~issue_title（冗余）~~，body（非冗余）; related_columns: action, *issue_id*, **body**, `issue_comment_id`, issue_comment_created_at, issue_comment_updated_at, issue_comment_author_association, *issue_comment_author_id*, *issue_comment_author_login*, issue_comment_author_type

'PullRequestEvent': ['issue_title', 'body'],  # content: issue_title（action = closed reopened labeled冗余，可只取opened）, body（action = closed reopened labeled冗余，可只取opened）; related_columns:id, type, action, actor_id, actor_login, *repo_id*, repo_name, org_id, org_login, created_at, `issue_id`, issue_number, **issue_title**, **body**, pull_commits, pull_additions, pull_deletions, pull_changed_files, pull_merged, *pull_merge_commit_sha*(PullRequestEvent 合入关闭时，或者repo发生PushEvent改变时，或者PullRequestReviewCommentEvent由作者以外的人员参与讨论之后，每次由任意人员created时，都会重新生成), pull_merged_at, pull_merged_by_id, pull_merged_by_login, pull_merged_by_type, pull_base_ref, pull_head_repo_id, pull_head_repo_name, pull_head_ref, [*repo_description*(repo级冗余), repo_size, repo_stargazers_count, repo_forks_count, repo_language, repo_has_issues, repo_has_projects, repo_has_downloads, repo_has_wiki, repo_has_pages, repo_license, repo_default_branch, repo_created_at, repo_updated_at, repo_pushed_at]

'PullRequestReviewEvent': ['body'],  # content: ~~issue_title（冗余）~~，body（非冗余）; related_columns: action, *issue_id*, **body**, *pull_merge_commit_sha*, pull_requested_reviewer_id, pull_requested_reviewer_login, pull_requested_reviewer_type, pull_review_comments, pull_review_state, pull_review_author_association, `pull_review_id`

'PullRequestReviewCommentEvent': ['body'],  # content: ~~issue_title（冗余）~~，body（非冗余）; related_columns: action, **body**, *pull_review_id*, `pull_review_comment_id`, *pull_review_comment_path*, pull_review_comment_position, *pull_review_comment_author_id*, *pull_review_comment_author_login*, pull_review_comment_author_type, pull_review_comment_author_association, pull_review_comment_created_at, pull_review_comment_updated_at

'PushEvent': ['push_commits.message'],   # content: push_commits.message（非冗余）; related_columns: action, actor_id（可能是bot，与push作者可能不同）, *actor_login*, `push_id`, push_size, push_distinct_size, push_ref, *push_head*, *push_commits.name*, *push_commits.email*(可通过name和email字段获取login和id 见https://www.coder.work/article/3180620), **push_commits.message**

'CommitCommentEvent': ['body'],  # content: body（非冗余）; related_columns: action, **body**, `commit_comment_id`, commit_comment_author_id, *commit_comment_author_login*, commit_comment_author_type, commit_comment_author_association, *commit_comment_path*, commit_comment_position, commit_comment_line, *commit_comment_sha*, commit_comment_created_at, commit_comment_updated_at, **注意commit_comment_sha下可以有多个评论的commit_comment_id, 其中commit_comment_sha主要来自PullRequestEvent的pull_merge_commit_sha和PushEvent的push_head，详见 https://github.com/X-lab2017/open-digger/issues/1268#issuecomment-1595314380 。**

'ReleaseEvent': ['release_body'],  # content: release_body（非冗余）; related_columns: action, `release_id`, *release_tag_name*, release_target_commitish, *release_name*, release_draft, release_author_id, release_author_login, release_author_type, release_prerelease, release_created_at, release_published_at, **release_body**, release_assets.name, [release_assets.uploader_login, release_assets.uploader_id, release_assets.content_type, release_assets.state, release_assets.size, release_assets.download_count]

~~'GollumEvent'(数据库中无body)~~: None, # content: None; related_columns: `id`, action, actor_id, *actor_login*, *gollum_pages.page_name*（gollum_pages.action = edited 冗余，可只取created）, gollum_pages.title, gollum_pages.action

~~'ForkEvent'(数据库中无body)~~: None, # content: None; related_columns: `id`, action, fork_forkee_id, *fork_forkee_full_name*, fork_forkee_owner_id, *fork_forkee_owner_login*, fork_forkee_owner_type

~~'MemberEvent'(数据库中无body)~~: None, # content: None; related_columns: `id`, action, actor_id, actor_login, repo_id, *repo_name*, org_id, org_login, member_id, *member_login*, member_type

~~'WatchEvent'(数据库中无body)~~: None, # content: None; related_columns: `id`, action, actor_id, *actor_login*, repo_id, *repo_name*

~~'PublicEvent'(数据库中无body)~~: None, # content: None; related_columns: `id`, action, actor_id, actor_login, repo_id, *repo_name*

~~'CreateEvent'(数据库中无body)~~: None, # content: None; related_columns: `id`, action, actor_id, actor_login, *create_ref*, create_ref_type, create_master_branch(repo级冗余), *create_description*(repo级冗余), create_pusher_type

~~'DeleteEvent'(数据库中无body)~~: None, # content: None; related_columns: `id`, action, actor_id, actor_login, *delete_ref*, delete_ref_type, delete_pusher_type

~~'PullRequestReviewThreadEvent'(数据库中无此特征)~~

~~'SponsorshipEvent'(数据库中无此特征)~~


# Event feature - simple

'IssuesEvent': ['issue_title', 'body'],  # content: issue_title（action = closed reopened labeled冗余，可只取opened）, body（action = closed reopened labeled冗余，可只取opened）; related_columns: id, type, action, actor_id, actor_login, *repo_id*, repo_name, org_id, org_login, created_at, `issue_id`, issue_number, **issue_title**, **body**, *issue_author_id*, *issue_author_login*, *issue_assignee_id*, *issue_assignee_login*, *issue_assignees.login*, *issue_assignees.id*

'IssueCommentEvent': ['body'],  # content: ~~issue_title（冗余）~~，body（非冗余）; related_columns: action, *issue_id*, **body**, `issue_comment_id`, *issue_comment_author_id*, *issue_comment_author_login*

'PullRequestEvent': ['issue_title', 'body']  # content: issue_title（action = closed reopened labeled冗余，可只取opened）, body（action = closed reopened labeled冗余，可只取opened）; related_columns: id, type, action, actor_id, actor_login, *repo_id*, repo_name, org_id, org_login, created_at, `issue_id`, issue_number, **issue_title**, **body**, *pull_merge_commit_sha*, pull_merged_by_id, pull_merged_by_login, pull_base_ref, pull_head_repo_id, pull_head_repo_name, pull_head_ref

'PullRequestReviewEvent': ['body'],  # content: ~~issue_title（冗余）~~，body（非冗余）; related_columns: action, *issue_id*, **body**, *pull_merge_commit_sha*, pull_requested_reviewer_id, pull_requested_reviewer_login, `pull_review_id`

'PullRequestReviewCommentEvent': ['body'],  # content: ~~issue_title（冗余）~~，body（非冗余）; related_columns: action, **body**, *pull_review_id*, `pull_review_comment_id`, *pull_review_comment_path*, *pull_review_comment_author_id*, *pull_review_comment_author_login*

'PushEvent': ['push_commits.message'],   # content: push_commits.message（非冗余）; related_columns: action, actor_id（可能是bot，与push作者可能不同）, *actor_login*, `push_id`, *push_head*, *push_commits.name*, *push_commits.email*(可通过name和email字段获取login和id 见https://www.coder.work/article/3180620), **push_commits.message**

'CommitCommentEvent': ['body'],  # content: body（非冗余）; related_columns: action, **body**, `commit_comment_id`, commit_comment_author_id, *commit_comment_author_login*, *commit_comment_path*, *commit_comment_sha*

'ReleaseEvent': ['release_body'],  # content: release_body（非冗余）; related_columns: action, `release_id`, *release_tag_name*, *release_name*, release_author_id, release_author_login, **release_body**, release_assets.name


----------------content 字段-------------------
## Collaboration

'IssuesEvent': ['issue_title', 'body'],  # content: issue_title（action = closed reopened labeled冗余，可只取opened），body（action = closed reopened labeled冗余，可只取opened）

'IssueCommentEvent': ['body'],  # content: body（非冗余）

'PullRequestEvent': ['issue_title', 'body'],  # content: issue_title（action = closed reopened labeled冗余，可只取opened）, body（action = closed reopened labeled冗余，可只取opened）

'PullRequestReviewEvent': ['body'],  # content: body（非冗余）

'PullRequestReviewCommentEvent': ['body'],  # content: body（非冗余）

'PushEvent': ['push_commits.message'],   # content: push_commits.message（非冗余）

'CommitCommentEvent': ['body'],  # content: body（非冗余）

## Distribution

'ReleaseEvent': ['release_body'],  # content: release_body（非冗余）


------------------分层----------------------

# 相关实体
1. 人

actor, org

2. 资源

project: repo, package, submodule

content: dir, file, text

3. **事件**

type: IssuesEvent, PullRequestEvent, PullRequestReviewEvent, IssueCommentEvent, PullRequestReviewCommentEvent, CommitCommentEvent, PushEvent, ReleaseEvent, GollumEvent, ForkEvent, MemberEvent, WatchEvent, PublicEvent, CreateEvent, DeleteEvent, ~PullRequestReviewThreadEvent(数据库中无此特征), SponsorshipEvent(数据库中无此特征)~

action: opened, closed, reopened, labeled, created

body: IssuesEvent issue_title body, IssueCommentEvent body, PullRequestEvent issue_title body, PullRequestReviewEvent body, PullRequestReviewCommentEvent body, PushEvent push_commits.message, CommitCommentEvent body, ReleaseEvent release_body


# 事件层次划分

github-event-types doc：https://docs.github.com/en/webhooks-and-events/events/github-event-types

1. Production Relations

  - 1.1. actor <--> repo
  
    actor -> actor: ~~Follow(人与人的社会信息网络，数据库中无此特征)~~;

    repo -> actor: MemberEvent, ~~SponsorshipEvent(数据库中无此特征)~~;

    actor -> repo: WatchEvent;

    repo -> repo: ForkEvent(主要关系属于repo->repo);

2. Product Operation:

  - 2.1. Repository层
    
    PublicEvent
    
  - 2.2. Branch层

    Branch::CreateEvent, Branch::DeleteEvent
  
  - 2.3. Commit层
  
    Tag::CreateEvent, Tag::DeleteEvent

3. **Development**

  - 3.1. **Collaboration**

    IssuesEvent, IssueCommentEvent, PullRequestEvent, PullRequestReviewEvent,  PullRequestReviewCommentEvent, ~~PullRequestReviewThreadEvent(数据库中无此特征)~~, PushEvent, CommitCommentEvent
    
  - 3.2. **Distribution**
    
    ReleaseEvent
    
  - 3.3. Crowdsourcing
  
    GollumEvent


# 链接类型

1. resouce and resoure

repo-repo: fork, submodule, import

repo-issue: open, close, reopen

issue-comment: create

comment-link: contain


2. actor and resource

repo-person: member

person-repo: maintain(create, delete), contribute(PullRequest, review), participate(IssueComment, open, close, reopen), star

3. actor and actor

org-org: follow

org-person: member, follow

person-person: follow


# 链接步长

1. 直接链接

repo-repo: fork, submodule, import

actor-repo: own, archive, star, fork

actor-actor: follow

repo-actor: member

2. 间接链接

repo-repo: same domain, same language, same functionality for a specific software

actor-repo: search, recommend

actor-actor: cooperate

'''

# 筛选与引用网络有关的属性
# https://github.com/X-lab2017/open-digger/blob/master/docs/assets/data_description.csv
columns_full = ['id', 'type', 'action', 'actor_id', 'actor_login', 'repo_id', 'repo_name', 'org_id', 'org_login', 'created_at', 'issue_id', 'issue_number', 'issue_title', 'body', 'issue_labels.name', 'issue_labels.color', 'issue_labels.default',
                'issue_labels.description', 'issue_author_id', 'issue_author_login', 'issue_author_type', 'issue_author_association', 'issue_assignee_id', 'issue_assignee_login', 'issue_assignees.login', 'issue_assignees.id', 'issue_created_at',
                'issue_updated_at', 'issue_comments', 'issue_closed_at', 'issue_comment_id', 'issue_comment_created_at', 'issue_comment_updated_at', 'issue_comment_author_association', 'issue_comment_author_id', 'issue_comment_author_login',
                'issue_comment_author_type', 'pull_commits', 'pull_additions', 'pull_deletions', 'pull_changed_files', 'pull_merged', 'pull_merge_commit_sha', 'pull_merged_at', 'pull_merged_by_id', 'pull_merged_by_login', 'pull_merged_by_type',
                'pull_requested_reviewer_id', 'pull_requested_reviewer_login', 'pull_requested_reviewer_type', 'pull_review_comments', 'pull_base_ref', 'pull_head_repo_id', 'pull_head_repo_name', 'pull_head_ref',
                'repo_description', 'repo_size', 'repo_stargazers_count', 'repo_forks_count', 'repo_language', 'repo_has_issues',
                'repo_has_projects', 'repo_has_downloads', 'repo_has_wiki', 'repo_has_pages', 'repo_license', 'repo_default_branch', 'repo_created_at', 'repo_updated_at', 'repo_pushed_at', 'pull_review_state', 'pull_review_author_association',
                'pull_review_id', 'pull_review_comment_id', 'pull_review_comment_path', 'pull_review_comment_position', 'pull_review_comment_author_id', 'pull_review_comment_author_login', 'pull_review_comment_author_type',
                'pull_review_comment_author_association', 'pull_review_comment_created_at', 'pull_review_comment_updated_at', 'push_id', 'push_size', 'push_distinct_size', 'push_ref', 'push_head', 'push_commits.name', 'push_commits.email',
                'push_commits.message', 'fork_forkee_id', 'fork_forkee_full_name', 'fork_forkee_owner_id', 'fork_forkee_owner_login', 'fork_forkee_owner_type', 'delete_ref', 'delete_ref_type', 'delete_pusher_type', 'create_ref', 'create_ref_type',
                'create_master_branch', 'create_description', 'create_pusher_type', 'gollum_pages.page_name', 'gollum_pages.title', 'gollum_pages.action', 'member_id', 'member_login', 'member_type', 'release_id', 'release_tag_name',
                'release_target_commitish', 'release_name', 'release_draft', 'release_author_id', 'release_author_login', 'release_author_type', 'release_prerelease', 'release_created_at', 'release_published_at', 'release_body',
                'release_assets.name', 'release_assets.uploader_login', 'release_assets.uploader_id', 'release_assets.content_type', 'release_assets.state', 'release_assets.size', 'release_assets.download_count', 'commit_comment_id',
                'commit_comment_author_id', 'commit_comment_author_login', 'commit_comment_author_type', 'commit_comment_author_association', 'commit_comment_path', 'commit_comment_position', 'commit_comment_line', 'commit_comment_sha',
                'commit_comment_created_at', 'commit_comment_updated_at']
columns_simple = ['id', 'type', 'action', 'actor_id', 'actor_login', 'repo_id', 'repo_name', 'org_id', 'org_login', 'created_at', 'issue_id', 'issue_number', 'issue_title', 'body', 'issue_author_id', 'issue_author_login', 'issue_assignee_id',
                  'issue_assignee_login', 'issue_assignees.login', 'issue_assignees.id', 'issue_comment_id', 'issue_comment_author_id', 'issue_comment_author_login', 'pull_merge_commit_sha', 'pull_merged_by_id', 'pull_merged_by_login',
                  'pull_requested_reviewer_id', 'pull_requested_reviewer_login', 'pull_base_ref', 'pull_head_repo_id', 'pull_head_repo_name', 'pull_head_ref', 'pull_review_id', 'pull_review_comment_id', 'pull_review_comment_path',
                  'pull_review_comment_author_id', 'pull_review_comment_author_login', 'push_id', 'push_head', 'push_commits.name', 'push_commits.email', 'push_commits.message', 'commit_comment_id', 'commit_comment_author_id', 'commit_comment_author_login',
                  'commit_comment_path', 'commit_comment_sha', 'release_id', 'release_tag_name', 'release_name', 'release_author_id', 'release_author_login', 'release_body', 'release_assets.name']

# 2.2.2.a 从种子数据的属性列中筛选出数据项中含有交叉引用的特征
# 筛选引用网络建立的描述基础
body_columns_dict = {
    'global_descriptions': ['repo_description'],
    'local_descriptions': [
        'issue_title',
        'body',  # fusion of issue_body, issue_comment_body, pull_review_comment_body, commit_comment_body
        # 'issue_body',  # IssueEvent 和 PullRequestEvent的内容
        #                 # body s.t. type='IssuesEvent' action in ['opened', 'closed', 'reopened', 'labeled']
        #                 # or body s.t. type='PullRequestEvent' action in ['opened', 'closed', 'reopened', 'labeled']
        # 'issue_comment_body',  # IssueCommentEvent的内容 # body s.t. type='IssueCommentEvent' action='created', 注意issue title保持所在的issue不变
        # 'pull_review_comment_body',  # PullRequestReviewCommentEvent的内容 # body s.t. type='PullRequestReviewCommentEvent' action='created'
        'push_commits.message',  # PushEvent
        # 'commit_comment_body',  # body s.t. type='CommitCommentEvent' action='added'
        'release_body',  # ReleaseEvent
    ]
}
event_columns_dict = {
    'basic': ['id', 'type', 'action', 'actor_id', 'actor_login', 'repo_id', 'repo_name', 'org_id', 'org_login', 'created_at'],
    'action_ref_entities': ['id', 'type', 'action', 'actor_id', 'actor_login', 'repo_id', 'repo_name', 'org_id', 'org_login', 'created_at', 'issue_id', 'issue_number', 'issue_comment_id', 'pull_merge_commit_sha', 'pull_review_id', 'pull_review_comment_id', 'push_id', 'push_head', 'commit_comment_id', 'commit_comment_sha', 'release_id']
    }

# SELECT distinct action FROM opensource.gh_events WHERE type='IssuesEvent';
# RESULT1: ['opened', 'closed', 'reopened', 'labeled']
# SELECT distinct action FROM opensource.gh_events WHERE type='PullRequestEvent';
# RESULT2: ['opened', 'closed', 'reopened', 'labeled']
# SELECT distinct action FROM opensource.gh_events WHERE type='IssueCommentEvent';
# RESULT3: ['created']
# SELECT distinct action FROM opensource.gh_events WHERE type='PullRequestReviewEvent';
# RESULT4: ['created']
# SELECT distinct action FROM opensource.gh_events WHERE type='PullRequestReviewCommentEvent';
# RESULT5: ['created']
# SELECT distinct action FROM opensource.gh_events WHERE type='CommitCommentEvent';
# RESULT6: ['added']


# 2.2.2.b 识别其中的交叉引用
# 定义正则模式
re_ref_patterns = {
    'Issue_PR': [r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/(?:issues|pull)/\d+#\w+-\d+(?![\d/])',  # Issue与PullRequest下的事件，包含了IssueEvent（示例：issues/{issue_number}#issue-{issue_id}）或PullRequestEvent事件（示例：(?:issues|pull)/{issue_number}#issue-{issue_id}），IssueCommentEvent事件（Issue与PullRequest下的评论均为IssueCommentEvent事件，示例{_repo_full_name}/issues/{issue_number}#issuecomment-{issue_comment_id}），PullRequestReview事件（示例：{issue_number}#pullrequestreview-{pull_review_id}）
                 r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/issues/\d+(?![\d/#])', r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/pull/\d+(?![\d/#])',  # IssueEvent与PullRequestEvent事件是两个独立的事件
                 r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/pull/\d+(?:#discussion_r|/files(?:/[0-9a-fA-F]{40})?#r)\d+(?![\d/])',  # PullRequestReveiwCommentEvent事件（示例：{issue_number}/files#r{pull_review_comment_id} or {issue_number}/files/{push_head}#r{pull_review_comment_id} or {issue_number}#discussion_r{pull_review_comment_id}）
                 r'(?<!\\)(?:(?!\\r|\\n|\\t)[^\s\(\[\]\>\'\";]?)*#0*[1-9][0-9]*(?![\d/#a-z])'],  # IssueEvent或PullRequestEvent事件
    'SHA': [r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/pull/\d+/commits/[0-9a-fA-F]{40}(?![0-9a-fA-F])', r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/commit/[0-9a-fA-F]{40}(?![0-9a-fA-F])', r'(?:(?<=[\s\(\[:])|(?<=^))([0-9a-fA-F]{40})(?![0-9a-fA-F])', r'(?:(?<=[\s\(\[:])|(?<=^))([0-9a-fA-F]{7})(?![0-9a-fA-F])'],  # 包含CommitEvent事件
    'Actor': [r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*(?![-A-Za-z0-9/])', r'(?:(?<=[\s\(\[])|(?<=^))(@[A-Za-z0-9][-0-9a-zA-Z]*)(?![-A-Za-z0-9/])', r'\w[-\w.+]*@(?:[A-Za-z0-9][-A-Za-z0-9]+\.)+[A-Za-z]{2,14}'],
    'Repo': [r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*(?![-A-Za-z0-9\./])'],
    'Branch': [r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/tree/[-0-9a-zA-Z]+(?![-A-Za-z0-9\./])'],
    'CommitComment': [r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/commit/[0-9a-fA-F]{40}#commitcomment-\d+(?![\d/])'],
    'Gollum': ['https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/wiki/[-_A-Za-z0-9\.%#/:]*(?![-_A-Za-z0-9\.%#/:])'],
    'Release': ['https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/releases/tag/[-_A-Za-z0-9\.%#/:]+(?![-_A-Za-z0-9\.%#/:])'],
    'GitHub_src': [r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/files[-_A-Za-z0-9\.%#/:]*(?![-_A-Za-z0-9\.%#/:])', r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/pull/\d+/files(?!#r)(?!/[0-9a-fA-F]{40}#r)[-_A-Za-z0-9\.%#/:]*(?![-_A-Za-z0-9\.%#/:])', r'https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/blob/[-_A-Za-z0-9\.%#/:]*(?![-_A-Za-z0-9\.%#/:])'],
    'GitHub_Other_Links': [r'(https?://(?:www\.)?github(?:-redirect\.dependabot)?\.com/[A-Za-z0-9][-0-9a-zA-Z]*/[A-Za-z0-9][-_0-9a-zA-Z\.]*/(?!issues|pull|commit|tree|wiki|releases|files|blob)[-_A-Za-z0-9\.%#/:]*(?![-_A-Za-z0-9\.%#/:]))'],
    'GitHub_Other_Service': [r'https?://(?!www\.)[-a-zA-Z]+\.github(?:-redirect\.dependabot)?\.com[-_A-Za-z0-9\.%#/:]*(?![-_A-Za-z0-9\.%#/:])'],
    'GitHub_Service_External_Links': [r'(https?://(?![-a-zA-Z]+\.github\.com|github\.com|github-redirect\.dependabot\.com)[-_A-Za-z0-9\.%#/:]*(?![-_A-Za-z0-9\.%#/:]))']
}


'''
# 数据层次结构
初始表：从github日志数据库github_log中的某一年度表中按项目名查询出的原始数据表。

基本表1：每个项目中每个record的每个字段、每个pattern类型的引用识别子串列表。基本表1可由初始表处理得到。

基本表2：每个项目中每个record的每个字段、每个pattern类型的引用识别子串去重后子串频数字典。基本表2可由基本表1处理得到。

## 数据总共有4层结构：repositories-records-record-regex_patterns
### 1. repositories: repo1, repo2, ... 

e.g. "elastic_elasticsearch"

以基本表1为数据集，可分析每个项目中每个record的引用频数，相关衍生表列举如下：
- 每个项目中每个record的每个字段、每个pattern识别出的引用子串数
- 每个项目中每个record的每个字段存在引用的频数及真值（任一pattern识别出子串即为真），真值表可用来过滤原始数据项
- 每个项目中每个record存在引用的频数及真值（即存在引用的record数，任一字段的任一pattern识别出子串即为真），真值表可用来过滤原始数据记录

### 2. records: record1, record2, ... 

e.g. id=14958930546

以基本表1为数据集，可分析每个record的引用频数，相关衍生表列举如下：
- 每个record的每个字段、每个pattern识别出的引用子串数
- 每个record的每个字段存在引用的频数及真值（任一pattern识别出子串即为真）
- 每个record存在引用的频数及真值（即存在引用的record数，任一字段的任一pattern识别出子串即为真）

### 3. record: issue_title, issue_body, issue_comment_body, pull_review_comment_body, ...

e.g. "	id	type	action	actor_id	actor_login	repo_id	repo_name	repo_description	org_id	org_login	created_at	..."
    "0	14958930546	WatchEvent	started	49124363	LL08	507775	elastic/elasticsearch	NaN	6764390	elastic	2021-01-27 01:56:18	..."
    
以基本表1为数据集，可分析每个msg_body字段的引用频数，相关衍生表列举如下：
- 每个msg_body字段、每个pattern识别出的引用子串数
- 每个msg_body字段存在引用的频数及真值（任一pattern识别出子串即为真）

### 4. regex_patterns: 'Issue_PR', 'SHA', 'actor', 'repo', 'github_src', 'github_other_links', 'outer_http'

e.g. Issue_PR 0 s1_subs: ['https://github.com/X-lab2017/open-research/issues/123#issue-1406887967'], SHA 0 s2_subs: ['https://github.com/X-lab2017/open-galaxy/pull/2/commits/7f9f3706abc7b5a9ad37470519f5066119ba46c2'], actor 0 s3_subs: ['https://github.com/birdflyi'], repo 0 s4_subs: ['https://github.com/X-lab2017/open-research'], ...

对于每个msg_body字段对应的数据项，每种pattern类型可识别的引用子串数（每一个被识别的引用都被记录下来，可看做命名实体，一个字符串被某一个pattern类型的某一个正则表达式所识别出的引用子串存储在一个列表中，属于同一个pattern类型的正则表达式的匹配结果将被不去重地合并到一整个列表中，表示一个字符串被某一个pattern类型所识别出的引用子串列表）：子串列表取长度
- 每个msg_body字段对应的数据项被每种pattern类型可识别的引用子串数
- 每个msg_body字段对应的数据项存在被每种pattern类型识别的频数及真值（pattern类型中的任一表达式识别出子串即为真）

'''

USE_RAW_STR = 0
USE_REG_SUB_STRS = 1
USE_REG_SUB_STRS_LEN = 2
use_data_confs = [USE_RAW_STR, USE_REG_SUB_STRS, USE_REG_SUB_STRS_LEN]
default_use_data_conf = use_data_confs[1]
