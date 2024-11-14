#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/14 6:27
# @Author : 'Lou Zehua'
# @File   : build_Graph.py

import networkx as nx
import numpy as np
import pandas as pd


def within_keys(k, d):
    if type(d) is dict:
        if k in d.keys():
            return True
    return False


def build_MultiDiGraph(df_src_tar, base_graph=None, src_node_attrs=None, tar_node_attrs=None, default_node_weight=1,
                       init_node_weight=False, default_node_type=None, node_type_canopy=False,
                       edge_attrs=None, default_edge_weight=1, init_edge_weight=True):
    '''
    :param df_src_tar: type pd.DataFrame, a dataframe with source node column and target node column.
    :param base_graph: a base graph can be parsed into nx.MultiGraph type
    :param src_node_attrs: set attributes to source nodes.
    :param tar_node_attrs: set attributes to target nodes.
    :param default_node_weight: 1
    :param init_node_weight: set default node weight to all the node attributes without the key "weight".
    :param default_node_type: set default node type when there is no "node_type" key in node_attrs.
        None for no default_node_type, the key "node_type" will not be generated automatically.
        default_node_type='__node_repr__' use node name as default node type.
        default_node_type='__column_name__' use the column name of df_src_tar as default node type.
    :param node_type_canopy: set node_type_canopy=True if you want to use bool flag dict as "node_type" to
        represent multi-types.
    :param edge_attrs:
        type pd.Series: set an edge attribute dict to each edge (u, v);
        type dict or None: set the parameter value of edge_attrs to all edges.
    :param default_edge_weight: 1
    :param init_edge_weight: set default edge weight to edge attributes without the key "weight".
    :return: MDG
    '''
    MDG = nx.MultiDiGraph(base_graph) if base_graph is not None else nx.MultiDiGraph()
    node_colname_pair = list(df_src_tar.columns)[0:2]
    src_node_list = df_src_tar[node_colname_pair[0]].values
    tar_node_list = df_src_tar[node_colname_pair[1]].values
    df_src_default_node_type = None
    df_tar_default_node_type = None
    if default_node_type is None:
        pass
    elif default_node_type == '__node_repr__':
        src_node_type_list = [str(n) for n in src_node_list]
        tar_node_type_list = [str(n) for n in tar_node_list]
        df_src_default_node_type = pd.DataFrame(np.array([src_node_list, src_node_type_list]).T)
        df_tar_default_node_type = pd.DataFrame(np.array([tar_node_list, tar_node_type_list]).T)
    elif default_node_type == '__column_name__':
        src_node_type_list = [node_colname_pair[0]] * len(src_node_list)
        tar_node_type_list = [node_colname_pair[1]] * len(tar_node_list)
        df_src_default_node_type = pd.DataFrame(np.array([src_node_list, src_node_type_list]).T)
        df_tar_default_node_type = pd.DataFrame(np.array([tar_node_list, tar_node_type_list]).T)
    else:
        raise ValueError("default_node_type must be in [None, '__node_repr__', '__column_name__']!")
    uv_list = list(df_src_tar[node_colname_pair].values)
    MDG = _build_MDG_nodes(MDG, src_node_list, src_node_attrs, default_node_weight, init_node_weight,
                           df_src_default_node_type, node_type_canopy)
    MDG = _build_MDG_nodes(MDG, tar_node_list, tar_node_attrs, default_node_weight, init_node_weight,
                           df_tar_default_node_type, node_type_canopy)
    MDG = _build_MDG_edges(MDG, uv_list, edge_attrs, default_edge_weight, init_edge_weight)
    return MDG


def _build_MDG_edges(MDG, uv_list, edge_attrs, default_edge_weight, init_edge_weight):
    if type(edge_attrs) is pd.Series:
        edge_attrs = pd.DataFrame(edge_attrs)
    if type(edge_attrs) is pd.DataFrame:  # 逐个遍历更新边的属性，可保留原本的边属性
        if len(uv_list) != len(edge_attrs):
            raise ValueError("The type of edge_attrs is DataFrame, but it has different length with df_src_tar!")
        edge_attrs_record_list = edge_attrs.to_dict('records')
        for uv, e_attr in zip(uv_list, edge_attrs_record_list):
            u, v = tuple(uv)
            if init_edge_weight and not within_keys("weight", e_attr):
                e_attr["weight"] = default_edge_weight
            MDG.add_edge(u, v, **e_attr)
    elif type(edge_attrs) is dict:  # 批量重置边的属性为传入的edge_attrs
        if init_edge_weight and not within_keys("weight", edge_attrs):
            edge_attrs["weight"] = default_edge_weight
        MDG.add_edges_from(uv_list, **edge_attrs)
    elif edge_attrs is None:  # 批量重置边的属性为默认的边属性
        if init_edge_weight:
            edge_attrs = {"weight": default_edge_weight}
        else:
            edge_attrs = {}
        MDG.add_edges_from(uv_list, **edge_attrs)
    else:
        raise TypeError("Unexpected edge_attrs type! It must be in [None, dict, pandas.Series, pandas.DataFrame]!")
    return MDG


def _build_MDG_nodes(MDG, node_list, node_attrs, default_node_weight, init_node_weight,
                     df_default_node_type: pd.DataFrame or None, node_type_canopy):
    if type(node_attrs) is pd.Series:
        node_attrs = pd.DataFrame(node_attrs)

    if df_default_node_type is None:
        default_node_type__nt_list = [None] * len(node_list)
    else:
        default_node_type__n_list = list(df_default_node_type[df_default_node_type.columns[0]].values)
        default_node_type__nt_list = list(df_default_node_type[df_default_node_type.columns[1]].values)
        assert (list(default_node_type__n_list) == list(node_list))

    if type(node_attrs) is pd.DataFrame:  # 逐个遍历更新节点的属性，可保留原本的节点属性
        if len(node_list) != len(node_attrs):
            raise ValueError("The type of node_attrs is DataFrame, but it has different length with node_list!")
        node_attrs_record_list = node_attrs.to_dict('records')
        for i in range(len(node_list)):
            n, n_attr, n_type = node_list[i], node_attrs_record_list[i], default_node_type__nt_list[i]
            if init_node_weight and not within_keys("weight", n_attr):
                n_attr["weight"] = default_node_weight
            n_attr = update_node_attrs_by_canopy_setting(n_type, n_attr, node_type_canopy)
            MDG.add_node(n, **n_attr)
    elif type(node_attrs) is dict:  # 批量重置节点的属性为传入的节点属性
        if init_node_weight and not within_keys("weight", node_attrs):
            node_attrs["weight"] = default_node_weight
        for i in range(len(node_list)):
            n, n_attr, n_type = node_list[i], node_attrs.copy(), default_node_type__nt_list[i]
            n_attr = update_node_attrs_by_canopy_setting(n_type, n_attr, node_type_canopy)
            MDG.add_node(n, **n_attr)
    elif node_attrs is None:  # 批量重置节点的属性为默认的节点属性
        if init_node_weight:
            node_attrs = {"weight": default_node_weight}
        else:
            node_attrs = {}
        for i in range(len(node_list)):
            n, n_attr, n_type = node_list[i], node_attrs.copy(), default_node_type__nt_list[i]
            n_attr = update_node_attrs_by_canopy_setting(n_type, n_attr, node_type_canopy)
            MDG.add_node(n, **n_attr)
    else:
        raise TypeError("Unexpected src_node_attrs type! It must be in [None, dict, pandas.Series, pandas.DataFrame]!")
    return MDG


def update_node_attrs_by_canopy_setting(n_type: str or None, n_attrs: dict, node_type_canopy: bool):
    node_type = {} if node_type_canopy else ""  # node_type_canopy = True时用字典表征多标签，用True和False的dict类型表征标签以方便更新标签状态
    if not within_keys("node_type", n_attrs) and n_type is not None:
        if node_type_canopy:
            node_type.update({n_type: True})
        else:
            node_type = n_type
        n_attrs["node_type"] = node_type
    elif node_type_canopy:
        raise ValueError(
            "The node_type_canopy cannot be True when node_type in the keys of src_node_attrs or tar_node_types.")
    return n_attrs


def set_node_type(G, new_nodes_type, mode="dict"):
    if mode == "list":
        if len(new_nodes_type) != len(G.nodes()):
            print("Error: the new_nodes_type has different length with G.nodes()!")
            return
        for i, v in enumerate(G.nodes()):
            G.nodes[v]['node_type'] = new_nodes_type[i]
    elif mode == "dict":
        for node, node_type in dict(new_nodes_type).items():
            G.nodes[node]['node_type'] = node_type
    elif mode == "DataFrame":
        for record in pd.DataFrame(new_nodes_type).to_dict("records"):
            node, node_type = list(record.values())[0:2]
            G.nodes[node]['node_type'] = node_type
    return G


def reset_node_type(G, reset_as='__node_repr__', apply_filter=None):
    if apply_filter is None:
        apply_filter = lambda x: True
    if reset_as is None:
        return G
    elif reset_as == '__node_repr__':  # 重置为每个结点的表征值，即每个结点单独一个类型
        for i, v in enumerate(G.nodes()):
            if apply_filter(G.nodes[v]):
                G.nodes[v]['node_type'] = str(v)
    else:  # 统一重置为传入的值
        for i, v in enumerate(G.nodes()):
            if apply_filter(G.nodes[v]):
                G.nodes[v]['node_type'] = str(reset_as)
    return G


def ch_node_type(G, old_node_type, new_node_type, match_func=None):
    '''
    # old_node_type can be a str or a dict, e.g.:
    #   str: old_node_type = 'node_type1'
    #   dict: old_node_type = {'node_type1':True, 'node_type2':False}
    '''
    if match_func is None:
        match_func = lambda x, y: x == y
    for i, v in enumerate(G.nodes()):
        if match_func(old_node_type, G.nodes[v]['node_type']):
            G.nodes[v]['node_type'] = new_node_type
    return G


def Graph_edge_filter(G, w_trunc=None):
    if not isinstance(G, nx.Graph):
        return

    if w_trunc is not None:
        if not isinstance(G, nx.MultiGraph):
            rm_edges = [(src, tar) for src, tar, w in G.edges.data("weight") if w < w_trunc]
        else:
            rm_edges = [(src, tar, key) for src, tar, key, w in G.edges.data(keys=True, data="weight") if w < w_trunc]
        G.remove_edges_from(rm_edges)
    return G


def MDG2DG(MDG, multiplicity=True, default_weight=1, default_multiplicity=1):
    if type(MDG) is not nx.MultiDiGraph:
        print('The type of MDG is not MultiDiGraph!')
        return None

    DG = nx.DiGraph()
    DG.add_nodes_from(MDG.nodes(data=True))
    for u, v, e_dict in MDG.edges(data=True):
        if 'weight' not in e_dict.keys():
            e_dict['weight'] = default_weight
        if multiplicity and 'multiplicity' not in e_dict.keys():
            e_dict['multiplicity'] = default_multiplicity
        if not DG.has_edge(u, v):
            DG.add_edge(u, v, **e_dict)
        else:
            e_dict['weight'] = DG.edges[u, v]['weight'] + e_dict['weight']
            if multiplicity:
                e_dict['multiplicity'] = int(DG.edges[u, v]['multiplicity']) + int(e_dict['multiplicity'])
            DG.edges[u, v].update(e_dict)
    return DG


def DG2G(DG, only_upper_triangle=False, multiplicity=True, double_self_loop=True, default_weight=1,
         default_multiplicity=1):
    if type(DG) is not nx.DiGraph:
        print('The type of DG is not DiGraph!')
        return

    if only_upper_triangle:
        print(
            'Warning: messages in the lower triangular will be lost! multiplicity and double_self_loop will be reset to False!')
        return nx.to_undirected(DG)

    G = nx.Graph()
    G.add_nodes_from(DG.nodes(data=True))

    for u, v, e_dict in DG.edges(data=True):
        if 'weight' not in e_dict.keys():
            e_dict['weight'] = default_weight
        if multiplicity and 'multiplicity' not in e_dict.keys():  # 多重有向图的边默认重数为1
            e_dict['multiplicity'] = default_multiplicity
        if not G.has_edge(u, v):
            G.add_edge(u, v, **e_dict)
        else:
            e_dict['weight'] = G.edges[u, v]['weight'] + e_dict['weight']  # 无向图G.edges[u, v]与G.edges[v, u]均指向上三角的相应坐标
            if multiplicity:
                e_dict['multiplicity'] = int(G.edges[u, v]['multiplicity']) + int(e_dict['multiplicity'])
            G.edges[u, v].update(e_dict)

    if double_self_loop:
        for u in G.nodes():
            if (u, u) in G.edges:
                G.edges[u, u]['weight'] = G.edges[u, u]['weight'] * 2
                if multiplicity:
                    G.edges[u, u]['multiplicity'] = int(G.edges[u, u]['multiplicity']) * 2
            else:
                pass
    return G


def MDG2G(MDG, multiplicity=True, double_self_loop=True, default_weight=1, default_multiplicity=1):
    DG = MDG2DG(MDG, multiplicity=multiplicity, default_weight=default_weight, default_multiplicity=default_multiplicity)
    G = DG2G(DG, only_upper_triangle=False, multiplicity=multiplicity, double_self_loop=double_self_loop, default_weight=default_weight, default_multiplicity=default_multiplicity)
    return G


def build_Graph(df_node_pair, base_graph=None, default_node_type='__column_name__', node_type_canopy=False,
                edge_attrs=None, default_edge_weight=1, init_edge_weight=True, w_trunc=1, out_g_type='G', **kwargs):
    MDG = build_MultiDiGraph(df_node_pair, base_graph=base_graph, src_node_attrs=kwargs.get("src_node_attrs"),
                             tar_node_attrs=kwargs.get("tar_node_attrs"),
                             default_node_weight=kwargs.get("default_node_weight", 1),
                             init_node_weight=kwargs.get("init_node_weight", False),
                             default_node_type=default_node_type, node_type_canopy=node_type_canopy,
                             edge_attrs=edge_attrs, default_edge_weight=default_edge_weight,
                             init_edge_weight=init_edge_weight)
    if out_g_type == 'MDG':
        G = MDG
    elif out_g_type == 'DG':
        G = MDG2DG(MDG, multiplicity=True, default_weight=default_edge_weight, default_multiplicity=1)
    elif out_g_type == 'G':
        double_self_loop = kwargs.get("double_self_loop", None)
        if double_self_loop is None:
            if type(MDG) in [nx.MultiDiGraph, nx.DiGraph]:
                default_double_self_loop = True
                print("Convert Directed Graph to undirected graph, set the default value of double_self_loop to True!")
            else:
                default_double_self_loop = False
            double_self_loop = default_double_self_loop
        G = MDG2G(MDG, multiplicity=True, double_self_loop=double_self_loop, default_weight=default_edge_weight, default_multiplicity=1)
    else:  # 默认不作任何处理
        raise ValueError("The value of out_g_type must be in ['MDG', 'DG', 'G']!")
    G = Graph_edge_filter(G, w_trunc=w_trunc)
    return G


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt  # matplotlib.use必须在本句执行前运行

    # build pattern test
    HIN_pattern = {
        "src_node_names": ['actor', 'issue', 'pr', 'repo', 'org'],  # row_names
        "tar_node_names": ['actor', 'issue', 'pr', 'repo', 'org'],  # col_names
        "pattern_ajacent_matrix": [[0, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]]
    }
    df_g_pattern = pd.DataFrame(np.array(HIN_pattern["pattern_ajacent_matrix"]), columns=HIN_pattern["src_node_names"],
                                index=HIN_pattern["tar_node_names"])
    triples = []
    for t_type, t_vec in df_g_pattern.items():
        for s_type, s_t_w in t_vec.items():
            triples.append([s_type, t_type, s_t_w])
    df_g_pattern_triples = pd.DataFrame(np.array(triples), columns=['src_node_type', 'tar_node_type', 'weight'])

    # MDG_pattern = build_MultiDiGraph(df_g_pattern_triples[['src_node_type', 'tar_node_type']], base_graph=None,
    #                                  src_node_attrs=pd.Series(df_g_pattern_triples.index, name="src_index"),
    #                                  tar_node_attrs=pd.Series(df_g_pattern_triples.index, name="tar_index"),
    #                                  default_node_weight=1, init_node_weight=False, default_node_type="__node_repr__",
    #                                  node_type_canopy=False, edge_attrs=df_g_pattern_triples[['weight']].astype(int),
    #                                  default_edge_weight=1, init_edge_weight=True)
    # # df_node_type_settings = pd.DataFrame(np.array([HIN_pattern["src_node_names"], HIN_pattern["src_node_names"]]).T,
    # #                                      columns=["node", "node_type"])
    # # MDG_pattern = set_node_type(MDG_pattern, df_node_type_settings, mode="DataFrame")
    # G_pattern = MDG2DG(MDG_pattern)
    # G_pattern = Graph_edge_filter(G_pattern, w_trunc=1)  # 截取权重不为0的边

    G_pattern = build_Graph(df_g_pattern_triples[['src_node_type', 'tar_node_type']],
                            default_node_type='__node_repr__', node_type_canopy=False,
                            edge_attrs=df_g_pattern_triples[['weight']].astype(int), default_weight=1, w_trunc=1,
                            out_g_type='DG')

    for n in G_pattern.nodes(data=True):
        print(n)
    for e in G_pattern.edges(data=True):
        print(e)

    pos = nx.circular_layout(G_pattern)

    color_map = {"actor": '#4B0082', 'issue': '#7CFC00', 'pr': '#800000', 'repo': '#FFA500', 'org': 'red'}
    node_size_map = {"actor": 500, 'issue': 500, 'pr': 500, 'repo': 500, 'org': 500}
    df_nodes_data = pd.DataFrame(dict(G_pattern.nodes(data=True)))
    df_nodes_type = df_nodes_data.loc["node_type"]
    node_color = df_nodes_type.apply(lambda x: color_map[x]).values
    node_size = df_nodes_type.apply(lambda x: node_size_map[x]).values
    node_labels = nx.get_node_attributes(G_pattern, 'node_type')
    edge_labels = nx.get_edge_attributes(G_pattern, 'weight')

    nx.draw_networkx_edge_labels(G_pattern, pos, edge_labels=edge_labels)
    nx.draw(G_pattern, pos, labels=node_labels, node_size=node_size, node_color=node_color, edge_color="black")

    plt.title('Graph Pattern', fontsize=15)
    # plt.savefig("HIN_pattern_tensorflow_scale1_trunc1.png", format="PNG")
    plt.show()
