#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/15 16:06
# @Author : 'Lou Zehua'
# @File   : migrate_in_cypher.py


# neo4j import
def get_Cypher_node_statement(node_var, node_prop_dict):
    # e.g. `Actor`{`node_type`: 'Actor'}
    node_var_statement = f"`{node_var}`"
    node_prop_statement = ', '.join(
        [f"`{node_prop_key}`: '{node_prop_value}'" for node_prop_key, node_prop_value in node_prop_dict.items()])
    node_statement = f"{node_var_statement} {{{node_prop_statement}}}"
    return node_statement


def get_Cypher_edge_statement(edge_var, edge_prop_dict):
    # e.g. `IssuesEvent::closed`{`event_type`:'IssuesEvent::closed', `weight`:'1'}
    edge_var_statement = f"`{edge_var}`"
    edge_prop_statement = ', '.join(
        [f"`{edge_prop_key}`: '{edge_prop_value}'" for edge_prop_key, edge_prop_value in edge_prop_dict.items()])
    edge_statement = f"{edge_var_statement} {{{edge_prop_statement}}}"
    return edge_statement


def get_neo4j_Cypher_import_statement_from_G(G, out_type=str, r_edge_var_key='event_type'):
    create_node_statements = []
    # print(f"CREATE(n:`{node_var}`{{`{list(node_prop_dict.keys())[0]}`: '{list(node_prop_dict.values())[0]}'}});")
    for n in G.nodes(data=True):
        node_var = n[0]
        node_prop_dict = n[1]
        node_statement = get_Cypher_node_statement(node_var, node_prop_dict)
        create_node_statement = f"CREATE (n:{node_statement});"
        create_node_statements.append(create_node_statement)

    create_edge_statements = []
    # print(f"MATCH (x:`{e[0]}`{{`node_type`:'{e[0]}'}}), (y:`{e[1]}`{{`node_type`:'{e[1]}'}})  \
    #     CREATE (x) -[r:`{list(e[2].values())[0]}`{{`{list(e[2].keys())[0]}`:'{list(e[2].values())[0]}', `{list(e[2].keys())[1]}`:'{list(e[2].values())[1]}'}}]-> (y);")
    for e in G.edges(data=True):
        s_node_var = e[0]
        s_node_prop_dict = G.nodes[e[0]]
        s_node_statement = get_Cypher_node_statement(s_node_var, s_node_prop_dict)
        t_node_var = e[1]
        t_node_prop_dict = G.nodes[e[1]]
        t_node_statement = get_Cypher_node_statement(t_node_var, t_node_prop_dict)
        r_edge_var = e[2][r_edge_var_key]
        r_edge_prop_dict = e[2]
        r_edge_statement = get_Cypher_edge_statement(r_edge_var, r_edge_prop_dict)
        create_edge_statement = f"MATCH (x:{s_node_statement}), (y:{t_node_statement}) CREATE (x)-[r:{r_edge_statement}]->(y);"
        create_edge_statements.append(create_edge_statement)

    statements = create_node_statements + create_edge_statements
    if out_type == str:
        statements = '\n'.join(statements)
    return statements


if __name__ == '__main__':
    from script.complex_network_analysis.build_network.build_gh_collab_net_pattern import G_pattern

    # print(get_neo4j_Cypher_import_statement_from_G(G_pattern))

    import os

    import pandas as pd

    from GH_CoRE.working_flow import read_csvs
    from GH_CoRE.working_flow.query_OSDB_github_log import get_repo_name_fileformat, get_repo_year_filename

    from etc import filePathConf
    from script.complex_network_analysis.build_network.build_Graph import build_Graph

    repo_names = ["TuGraph-family/tugraph-db"][0:1]
    year = 2023
    relation_extraction_save_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                              'GitHub_Collaboration_Network_repos')
    filenames_exists = os.listdir(relation_extraction_save_dir)
    if repo_names:
        repo_names_fileformat = list(map(get_repo_name_fileformat, repo_names))
        filenames = [get_repo_year_filename(s, year) for s in repo_names_fileformat]
        filenames = [filename for filename in filenames if filename in filenames_exists]
    else:
        filenames = filenames_exists
    df_dbms_repos_dict = read_csvs(relation_extraction_save_dir, filenames=filenames, index_col=None)
    # test for 1 repo case
    repo_keys = list(df_dbms_repos_dict.keys())
    df_dbms_repo = df_dbms_repos_dict[repo_keys[0]]
    G_repo = build_Graph(df_dbms_repo, base_graph=None, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
                         src_node_attrs=None, tar_node_attrs=None,
                         init_node_weight=True, nt_key_in_attr="node_type",
                         default_node_types=['src_entity_type', 'tar_entity_type'], node_type_canopy=False,
                         edge_attrs=pd.Series(df_dbms_repo.to_dict("records")),
                         init_edge_weight=True, et_key_in_attr="edge_type", default_edge_type="event_type",
                         edge_type_canopy=True,
                         attrs_is_shared_key_pdSeries=False, use_df_col_as_default_type=True, w_trunc=1,
                         out_g_type='DG')
    print(get_neo4j_Cypher_import_statement_from_G(G_repo))
