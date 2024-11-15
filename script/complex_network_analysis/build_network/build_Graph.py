#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/14 6:27
# @Author : 'Lou Zehua'
# @File   : build_Graph.py

import re

import networkx as nx
import numpy as np
import pandas as pd


def within_keys(k, d):
    if type(d) is dict:
        if k in d.keys():
            return True
    return False


def build_MultiDiGraph(df_src_tar, src_tar_colnames=None, base_graph=None, src_node_attrs=None, tar_node_attrs=None, default_node_weight=1,
                       init_node_weight=False, nt_key_in_attr="node_type", default_node_types=None, node_type_canopy=False,
                       edge_attrs=None, default_edge_weight=1, init_edge_weight=False, et_key_in_attr="edge_type",
                       default_edge_type=None, attrs_is_shared_key_pdSeries=True, use_df_col_as_default_type=False):
    """
    :param df_src_tar: type pd.DataFrame, a dataframe within source node column and target node column.
    :param src_tar_colnames: the source node column name and target node column name in df_src_tar.columns
    :param base_graph: a base graph can be parsed into nx.MultiGraph type
    :param src_node_attrs: pd.Series or column_name, set attributes to source nodes.
    :param tar_node_attrs: pd.Series or column_name, set attributes to target nodes.
    :param default_node_weight: 1
    :param init_node_weight: set default node weight to all the node attributes without the key "weight".
    :param nt_key_in_attr: node type key in any src_node_attr and tar_node_attr
    :param default_node_types: set default node type when there is no "node_type" key in node_attrs.
        The "node_type" in src_node_attrs and tar_node_attrs always take effect first.
        default_edge_type=None for no default_node_types, the key "node_type" will not be generated automatically.
        default_node_types=[default_src_node_type_setting, default_tar_node_type_setting] the node type settings can be:
            "__repr__" use node name as default node type.
            name_str: if use_df_col_as_default_type=True, use the name_str as column_name of df_src_tar to get a series
                of default node type, column_name cannot be "__repr__"; else use the name_str as default node type.
    :param node_type_canopy: set node_type_canopy=True to use bool flag dict as "node_type" to represent multi-types.
    :param edge_attrs:
        type pd.Series: set an edge attribute dict to each edge (u, v);
        type dict or None: set the parameter value of edge_attrs to all edges.
    :param default_edge_weight: 1
    :param init_edge_weight: set default edge weight to edge attributes without the key "weight".
    :param et_key_in_attr: edge type key in any edge_attr
    :param default_edge_type: set default edge type when there is no "edge_type" key in edge_attrs.
        The "edge_type" in edge_attrs always take effect first.
        default_edge_type=None for no default_edge_type, the key "edge_type" will not be generated automatically.
        default_edge_type="__repr__" use SrcNodeType_TarNodeType pattern as default edge type for each edge.
        default_edge_type=name_str: if use_df_col_as_default_type=True, use the name_str as column_name of df_src_tar to
            get a series of default edge type, column_name cannot be "__repr__"; else use the name_str as default edge type.
    :param attrs_is_shared_key_pdSeries: the types of src_node_attrs, tar_node_attrs, edge_attrs are pd.Series if True
        else list(dict) or pd.Series(dict) records.
    :param use_df_col_as_default_type: if True, use default_edge_type or elements in default_node_types to get
        the type column in df_src_tar. However, a default settings should not be too complex.
    :return: MDG
    """
    MDG = nx.MultiDiGraph(base_graph) if base_graph is not None else nx.MultiDiGraph()
    if src_tar_colnames is not None:
        node_colname_pair = src_tar_colnames[0:2]
    else:
        node_colname_pair = list(df_src_tar.columns)[0:2]
    src_node_list = df_src_tar[node_colname_pair[0]].values
    tar_node_list = df_src_tar[node_colname_pair[1]].values
    uv_list = list(df_src_tar[node_colname_pair].values)
    if src_node_attrs is None:
        src_node_attrs = pd.Series([{}] * len(src_node_list))
    else:
        if isinstance(src_node_attrs, str) and src_node_attrs in df_src_tar.columns:
            try:
                src_node_attrs = df_src_tar[src_node_attrs]
            except Exception:
                raise ValueError("src_node_attrs must be a column name in df_src_tar.columns when src_node_attrs is a str!")
        elif len(src_node_attrs) == len(src_node_list) and all([isinstance(elem, dict) or pd.isna(elem)
                                                                for elem in src_node_attrs]):
            src_node_attrs = pd.Series(src_node_attrs)
        else:
            raise TypeError("Unexpected src_node_attrs type! The types of src_node_attrs, tar_node_attrs, "
                            "edge_attrs should be pd.Series or str or list(dict) records.")
        if attrs_is_shared_key_pdSeries:
            src_node_attrs = pd.Series(src_node_attrs.to_frame().to_dict("records"))

    if tar_node_attrs is None:
        tar_node_attrs = pd.Series([{}] * len(tar_node_list))
    else:
        if isinstance(tar_node_attrs, str) and tar_node_attrs in df_src_tar.columns:
            try:
                tar_node_attrs = df_src_tar[tar_node_attrs]
            except Exception:
                raise ValueError("tar_node_attrs must be a column name in df_src_tar.columns when tar_node_attrs is a str!")
        elif len(tar_node_attrs) == len(tar_node_list) and all([isinstance(elem, dict) or pd.isna(elem)
                                                                for elem in tar_node_attrs]):
            tar_node_attrs = pd.Series(tar_node_attrs)
        else:
            raise TypeError("Unexpected src_node_attrs type! The types of src_node_attrs, tar_node_attrs, "
                            "edge_attrs should be pd.Series or str or list(dict) records.")
        if attrs_is_shared_key_pdSeries:
            tar_node_attrs = pd.Series(tar_node_attrs.to_frame().to_dict("records"))

    if edge_attrs is None:
        edge_attrs = pd.Series([{}] * len(df_src_tar))
    else:
        if isinstance(edge_attrs, str) and edge_attrs in df_src_tar.columns:
            try:
                edge_attrs = df_src_tar[edge_attrs]
            except Exception:
                raise ValueError("edge_attrs must be a column name in df_src_tar.columns when edge_attrs is a str!")
        elif len(edge_attrs) == len(tar_node_list) and all(
                [isinstance(elem, dict) or pd.isna(elem) for elem in edge_attrs]):
            edge_attrs = pd.Series(edge_attrs)
        else:
            raise TypeError("Unexpected src_node_attrs type! The types of src_node_attrs, tar_node_attrs, "
                            "edge_attrs should be pd.Series or str or list(dict) records.")
        if attrs_is_shared_key_pdSeries:
            edge_attrs = pd.Series(edge_attrs.to_frame().to_dict("records"))

    # update ser_src_node_type and ser_tar_node_type by default value
    ser_src_node_type = None
    ser_tar_node_type = None
    ser_edge_type = None
    if default_node_types is None:
        pass
    elif isinstance(default_node_types, list) or isinstance(default_node_types, tuple):
        default_src_node_type, default_tar_node_type = list(default_node_types)[:2]
        if default_src_node_type is None:
            pass
        elif default_src_node_type == '__repr__':
            src_node_type_list = [str(n) for n in src_node_list]
            ser_src_node_type = pd.Series(src_node_type_list)
        elif isinstance(default_src_node_type, str):
            if use_df_col_as_default_type:
                type_column_name = default_src_node_type
                try:
                    src_node_type_list = df_src_tar[type_column_name].values.tolist()
                except Exception:
                    raise ValueError(f"The type name '{type_column_name}' should be in the "
                                     f"df_src_tar.columns when use_df_col_as_default_type=True!")
                ser_src_node_type = pd.Series(src_node_type_list)
            else:
                src_node_type_list = [default_src_node_type] * len(src_node_list)
                ser_src_node_type = pd.Series(src_node_type_list)
        else:
            raise ValueError("Each default_node_type must be in [None, '__repr__', name_str]!")

        if default_tar_node_type is None:
            pass
        elif default_tar_node_type == '__repr__':
            tar_node_type_list = [str(n) for n in tar_node_list]
            ser_tar_node_type = pd.Series(tar_node_type_list)
        elif isinstance(default_tar_node_type, str):
            if use_df_col_as_default_type:
                type_column_name = default_tar_node_type
                try:
                    tar_node_type_list = df_src_tar[type_column_name].values.tolist()
                except Exception:
                    raise ValueError(f"the type name '{type_column_name}' should be in the "
                                     f"df_src_tar.columns when use_df_col_as_default_type=True!")
                ser_tar_node_type = pd.Series(tar_node_type_list)
            else:
                tar_node_type_list = [default_tar_node_type] * len(tar_node_list)
                ser_tar_node_type = pd.Series(tar_node_type_list)
        else:
            raise ValueError("Each default_node_type must be in [None, '__repr__', name_str]!")

    # update ser_edge_type by default value
    if default_edge_type is None:
        pass
    elif default_edge_type == '__repr__':
        if ser_src_node_type is None:
            ser_src_node_type = pd.Series([None] * len(src_node_list))
        if ser_tar_node_type is None:
            ser_tar_node_type = pd.Series([None] * len(tar_node_list))
        assert(len(ser_src_node_type) == len(ser_tar_node_type) == len(df_src_tar))
        edge_type_list = [str(ser_src_node_type.iloc[i]) + '_' + str(ser_tar_node_type.iloc[i]) for i in range(len(df_src_tar))]
        ser_edge_type = pd.Series(edge_type_list)
    elif isinstance(default_edge_type, str):
        if use_df_col_as_default_type:
            type_column_name = default_edge_type
            try:
                edge_type_list = df_src_tar[type_column_name].values.tolist()
            except Exception:
                raise ValueError(f"the type name '{type_column_name}' should be in the "
                                 f"df_src_tar.columns when use_df_col_as_default_type=True!")
            ser_edge_type = pd.Series(edge_type_list)
        else:
            edge_type_list = [default_edge_type] * len(src_node_list)
            ser_edge_type = pd.Series(edge_type_list)
    else:
        raise ValueError("The default_edge_type must be in [None, '__repr__', name_str]!")

    # build MDG
    MDG = _build_MDG_nodes(MDG, src_node_list, src_node_attrs, default_node_weight, init_node_weight, ser_src_node_type,
                           node_type_canopy, nt_key_in_attr)
    MDG = _build_MDG_nodes(MDG, tar_node_list, tar_node_attrs, default_node_weight, init_node_weight, ser_tar_node_type,
                           node_type_canopy, nt_key_in_attr)
    MDG = _build_MDG_edges(MDG, uv_list, edge_attrs, default_edge_weight, init_edge_weight, ser_edge_type,
                           et_key_in_attr)
    return MDG


def _build_MDG_nodes(MDG, node_list, node_attrs, default_node_weight, init_node_weight,
                     ser_node_type: pd.Series or None, node_type_canopy, nt_key_in_attr="node_type"):
    try:
        node_attrs = pd.Series(node_attrs) if node_attrs is not None else pd.Series([{}] * len(node_list))
        ser_node_type = pd.Series(ser_node_type) if ser_node_type is not None else pd.Series([None] * len(node_list))
    except TypeError:
        raise TypeError("Unexpected types of node_attrs or ser_node_type! They must be in [None, pandas.Series]!")
    assert(len(node_list) == len(node_attrs) == len(ser_node_type))
    # 逐个遍历更新节点的属性，可保留原本的节点属性
    for i in range(len(node_list)):
        n, n_attr, n_type = node_list[i], node_attrs.iloc[i], ser_node_type.iloc[i]
        n_attr = dict(n_attr) if n_attr is not None else {}
        if n_type is not None:
            n_type = str(n_type)
            n_attr = update_node_attr_by_canopy_setting(n_attr, n_type, node_type_canopy, nt_key_in_attr)
        if init_node_weight and not within_keys("weight", n_attr):
            n_attr["weight"] = default_node_weight
        MDG.add_node(n, **n_attr)
    return MDG


def update_node_attr_by_canopy_setting(n_attr: dict, n_type: str or None, node_type_canopy: bool, nt_key_in_attr: str):
    nt_key_in_attr = nt_key_in_attr if nt_key_in_attr is not None else "node_type"
    if n_type is not None:
        old_node_type = n_attr.get(nt_key_in_attr, None)
        # node_type_canopy = True时用字典表征多标签，用True和False的dict类型表征标签以方便更新标签状态
        if node_type_canopy:
            node_type = dict(old_node_type) if old_node_type is not None else {}
            node_type.update({n_type: True})  # update
        else:
            node_type = str(old_node_type) if old_node_type is not None else ""
            node_type = n_type or node_type  # update
        n_attr[nt_key_in_attr] = node_type
    return n_attr


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


def reset_node_type(G, reset_as='__repr__', apply_filter=None):
    if apply_filter is None:
        apply_filter = lambda x: True
    if reset_as is None:
        return G
    elif reset_as == '__repr__':  # 重置为每个结点的表征值，即每个结点单独一个类型
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


def _build_MDG_edges(MDG, uv_list, edge_attrs, default_edge_weight, init_edge_weight, ser_edge_type: pd.Series or None,
                     et_key_in_attr="edge_type"):
    try:
        edge_attrs = pd.Series(edge_attrs) if edge_attrs is not None else pd.Series([{}] * len(uv_list))
        ser_edge_type = pd.Series(ser_edge_type) if ser_edge_type is not None else pd.Series([None] * len(uv_list))
    except TypeError:
        raise TypeError("Unexpected types of edge_attrs or ser_edge_type! They must be in [None, pandas.Series]!")
    assert(len(uv_list) == len(edge_attrs) == len(ser_edge_type))

    for i in range(len(uv_list)):
        uv, e_attr, e_type = uv_list[i], edge_attrs.iloc[i], ser_edge_type.iloc[i]
        u, v = tuple(uv)
        e_attr = dict(e_attr) if e_attr is not None else {}
        if e_type is not None:
            e_type = str(e_type)
            old_edge_type = e_attr.get(et_key_in_attr, None)
            edge_type = str(old_edge_type) if old_edge_type is not None else ""
            edge_type = e_type or edge_type  # update
            e_attr[et_key_in_attr] = edge_type
        if init_edge_weight and not within_keys("weight", e_attr):
            e_attr["weight"] = default_edge_weight
        MDG.add_edge(u, v, **e_attr)
    return MDG


def update_edge_attr_by_canopy_setting(e_attr: dict or str, e_type: dict or str or None, edge_type_canopy: bool,
                                       et_key_in_attr: str):

    et_key_in_attr = et_key_in_attr if et_key_in_attr is not None else "edge_type"
    if e_type is not None:
        old_edge_type = e_attr.get(et_key_in_attr, None)
        if edge_type_canopy:
            if type(old_edge_type) is str:
                old_edge_type = {old_edge_type: True}
            edge_type = dict(old_edge_type) if old_edge_type is not None else {}
            temp_to_update = {e_type: True} if type(e_type) is str else dict(e_type)
            edge_type.update(temp_to_update)  # update
        else:
            edge_type = str(old_edge_type) if old_edge_type is not None else None
            try:
                temp_to_update = e_type if type(e_type) is str else [k for k, v in dict(e_type).items() if v][0]
            except Exception:
                temp_to_update = None
            edge_type = temp_to_update or edge_type  # update
        e_attr[et_key_in_attr] = edge_type
    return e_attr


def MDG2DG(MDG, multiplicity=True, default_weight=1, default_multiplicity=1, edge_type_canopy=False,
           et_key_in_attr="edge_type"):
    """
    :param MDG:
    :param multiplicity:
    :param default_weight:
    :param default_multiplicity:
    :param edge_type_canopy: set edge_type_canopy=True to use bool flag dict as "edge_type" to represent multi-types.
    :param et_key_in_attr: edge type key in any edge_attr, only takes effect when edge_type_canopy=True.
    :return:
    """
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
            if edge_type_canopy:
                e_attr_old = DG.edges[u, v]
                e_type = e_dict.get(et_key_in_attr, None)
                e_attr = update_edge_attr_by_canopy_setting(e_attr_old, e_type, edge_type_canopy, et_key_in_attr)
                e_dict[et_key_in_attr] = e_attr[et_key_in_attr]
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


def MDG2G(MDG, multiplicity=True, double_self_loop=True, default_weight=1, default_multiplicity=1,
          edge_type_canopy=False, et_key_in_attr="edge_type"):
    DG = MDG2DG(MDG, multiplicity=multiplicity, default_weight=default_weight, default_multiplicity=default_multiplicity,
                edge_type_canopy=edge_type_canopy, et_key_in_attr=et_key_in_attr)
    G = DG2G(DG, only_upper_triangle=False, multiplicity=multiplicity, double_self_loop=double_self_loop,
             default_weight=default_weight, default_multiplicity=default_multiplicity)
    return G


def build_Graph(df_src_tar, src_tar_colnames=None, base_graph=None, src_node_attrs=None, tar_node_attrs=None, default_node_weight=1,
                init_node_weight=False, default_node_types=None, node_type_canopy=False, edge_attrs=None,
                default_edge_weight=1, init_edge_weight=False, default_edge_type=None, edge_type_canopy=False,
                attrs_is_shared_key_pdSeries=True, w_trunc=1, out_g_type='G', **kwargs):
    nt_key_in_attr = kwargs.get("nt_key_in_attr", "node_type")
    et_key_in_attr = kwargs.get("et_key_in_attr", "edge_type")
    use_df_col_as_default_type = kwargs.get("use_df_col_as_default_type", False)
    MDG = build_MultiDiGraph(df_src_tar, src_tar_colnames=src_tar_colnames, base_graph=base_graph, src_node_attrs=src_node_attrs,
                             tar_node_attrs=tar_node_attrs, default_node_weight=default_node_weight,
                             init_node_weight=init_node_weight, nt_key_in_attr=nt_key_in_attr,
                             default_node_types=default_node_types, node_type_canopy=node_type_canopy,
                             edge_attrs=edge_attrs, default_edge_weight=default_edge_weight,
                             init_edge_weight=init_edge_weight, et_key_in_attr=et_key_in_attr,
                             default_edge_type=default_edge_type, attrs_is_shared_key_pdSeries=attrs_is_shared_key_pdSeries,
                             use_df_col_as_default_type=use_df_col_as_default_type)
    if et_key_in_attr is None:
        if type(edge_attrs) is str:
            et_key_in_attr = str(edge_attrs)
        elif isinstance(edge_attrs, pd.Series):
            if pd.Series(edge_attrs).name is not None:
                et_key_in_attr = pd.Series(edge_attrs).name
    if out_g_type == 'MDG':
        G = MDG
    elif out_g_type == 'DG':
        G = MDG2DG(MDG, multiplicity=True, default_weight=default_edge_weight, default_multiplicity=1,
                   edge_type_canopy=edge_type_canopy, et_key_in_attr=et_key_in_attr)
    elif out_g_type == 'G':
        double_self_loop = kwargs.get("double_self_loop", None)
        if double_self_loop is None:
            if type(MDG) in [nx.MultiDiGraph, nx.DiGraph]:
                default_double_self_loop = True
                print("Convert Directed Graph to undirected graph, set the default value of double_self_loop to True!")
            else:
                default_double_self_loop = False
            double_self_loop = default_double_self_loop
        G = MDG2G(MDG, multiplicity=True, double_self_loop=double_self_loop, default_weight=default_edge_weight,
                  default_multiplicity=1, edge_type_canopy=edge_type_canopy, et_key_in_attr=et_key_in_attr)
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

    df_g_pattern_triples['weight'] = df_g_pattern_triples['weight'].astype(int)
    G_pattern = build_Graph(df_g_pattern_triples[['src_node_type', 'tar_node_type', 'weight']],
                            default_node_types=['__repr__', '__repr__'], node_type_canopy=False,
                            edge_attrs='weight', default_edge_weight=1, w_trunc=1, out_g_type='DG')
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
