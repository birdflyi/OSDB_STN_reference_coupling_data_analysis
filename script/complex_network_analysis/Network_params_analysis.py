#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/17 19:47
# @Author : 'Lou Zehua'
# @File   : Network_params_analysis.py
import copy
import os

import networkx as nx
import numpy as np
import pandas as pd
from GH_CoRE.working_flow import get_repo_name_fileformat, get_repo_year_filename, read_csvs
from matplotlib import pyplot as plt

from etc import filePathConf
from script.complex_network_analysis.build_network.build_Graph import build_Graph


def get_pattern_from_G(G):
    G_pattern = copy.deepcopy(G)
    for u, e_dict in G_pattern.nodes(data=True):
        e_dict['weight'] = 0
        e_dict['multiplicity'] = 0

    for u, v, e_dict in G_pattern.edges(data=True):
        e_dict['weight'] = 0
        e_dict['multiplicity'] = 0

    G = G_max_sub

    # init G.nodes
    for u, e_dict in G.nodes(data=True):
        e_dict['weight'] = 1
        e_dict['multiplicity'] = 1

    # count 'weight' and 'multiplicity'
    for u, e_dict in G.nodes(data=True):
        u_type = e_dict['node_type']
        G_pattern.nodes[u_type]['weight'] += e_dict['weight']
        G_pattern.nodes[u_type]['multiplicity'] += 1

    for u, v, e_dict in G.edges(data=True):
        u_type = G.nodes[u]['node_type']
        v_type = G.nodes[v]['node_type']
        if (u_type, v_type) not in G_pattern.edges:
            (u_type, v_type) = (v_type, u_type)
        if (u_type, v_type) in G_pattern.edges:
            G_pattern.edges[u_type, v_type]['weight'] += e_dict['weight']
            G_pattern.edges[u_type, v_type]['multiplicity'] += 1
        else:
            print("Error in <u, v> direction!")
            break

    for u, e_dict in G_pattern.nodes(data=True):
        print(G_pattern.nodes[u]['node_type'], e_dict)

    for u, v, e_dict in G_pattern.edges(data=True):
        print(G_pattern.nodes[u]['node_type'], G_pattern.nodes[v]['node_type'], e_dict)

    return G_pattern

# pos = nx.circular_layout(G_pattern)
#
# color_map = {"actor": '#4B0082', 'issue': '#7CFC00', 'pr': '#800000', 'repo': '#FFA500', 'org': 'red'}
# node_size_map = {"actor": 500, 'issue': 500, 'pr': 500, 'repo': 500, 'org': 500}
# node_color = []
# node_size = []
# for n, d in G_pattern.nodes(data=True):
#     node_color.append(color_map[d['node_type']])
#     node_size.append(node_size_map[d['node_type']])
#
# node_labels1 = nx.get_node_attributes(G_pattern, 'node_type')
# node_labels2 = nx.get_node_attributes(G_pattern, 'weight')
# edge_labels = nx.get_edge_attributes(G_pattern, 'multiplicity')
#
# node_labels = {}
# for k in node_labels1.keys():
#     node_labels[k] = str(node_labels1[k]) + ": " + str(node_labels2[k])
# nx.draw_networkx_edge_labels(G_pattern, pos, edge_labels=edge_labels)
# nx.draw(G_pattern, pos, labels=node_labels, node_size=node_size, node_color=node_color, edge_color="black")
#
# plt.title('Graph Pattern', fontsize=15)
# plt.savefig("HIN_pattern_tensorflow_scale1_trunc1.png", format="PNG")
# plt.show()

import logging
logger = logging.getLogger(__name__)

def read_graph(graph_path):
    file_format = graph_path.split(".")[-1]
    if file_format == "graphml":
        return nx.read_graphml(graph_path)
    elif file_format == "gml":
        return nx.read_gml(graph_path)
    elif file_format == "gexf":
        return nx.read_gexf(graph_path)
    elif file_format == "net":
        return nx.read_pajek(graph_path)
    elif file_format == "yaml":
        return nx.read_yaml(graph_path)
    elif file_format == "gpickle":
        return nx.read_gpickle(graph_path)
    else:
        logging.warning("File format not found, returning empty graph.")
    return nx.MultiDiGraph()


if __name__ == '__main__':
    repo_names = ["TuGraph-family/tugraph-db", "facebook/rocksdb", "cockroachdb/cockroach"][0:1]
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
                         init_node_weight=True, nt_key_in_attr="node_type", default_node_types=['src_entity_type', 'tar_entity_type'], node_type_canopy=False,
                         edge_attrs=pd.Series(df_dbms_repo.to_dict("records")),
                         init_edge_weight=True, et_key_in_attr="edge_type", default_edge_type="event_type", edge_type_canopy=True,
                         attrs_is_shared_key_pdSeries=False, use_df_col_as_default_type=True, w_trunc=1, out_g_type='G')
    G = G_repo
    lc = max(nx.connected_components(G), key=len)  # max connected components
    G_max_sub = G.subgraph(lc)
    print(f"Max subgraph nodes ratio: {len(lc)/len(G.nodes)}")
    print(f"G: Nodes: {len(G.nodes)}, Edges: {len(G.edges)}, Edge Density: {2*len(G.edges)/(len(G.nodes)*(len(G.nodes)-1))}")
    print(f"G_max_sub: Nodes: {len(G_max_sub.nodes)}, Edges: {len(G_max_sub.edges)}, Edge Density: {2*len(G_max_sub.edges)/(len(G_max_sub.nodes)*(len(G_max_sub.nodes)-1))}")

    # 宏观统计及zipf分布
    n = len(G.nodes())
    print('n = ', n)
    import math

    e_threshold = n * math.log(n)
    print('e_threshold = ', e_threshold)

    e = len(G.edges())
    print('e = ', e)
    print('G_max_sub is a sparse graph:', e < e_threshold)

    k_avg = 2 * e / n
    print("平均度: k_avg = ", k_avg)

    degree_dict = dict(nx.degree(G))
    degree_sequence = sorted(degree_dict.values(), reverse=True)  # degree sequence decrease order
    # print(degree_sequence)
    dmax = max(degree_sequence)
    # print(degree_sequence)
    plt.loglog(np.array(range(len(degree_sequence))) + 1, degree_sequence, 'b-', marker='o')
    plt.title("Degree Rank Chart")
    plt.ylabel("Degree")
    plt.xlabel("Rank")

    plt.savefig("G_Degree_Rank_loglog_zipf.png")
    plt.show()

    # 幂律分布
    import numpy as np

    n = len(degree_sequence)
    degree_set = set(degree_sequence)
    k_list = sorted(list(degree_set))

    vals = np.zeros(len(k_list))
    k_freqNum_dict = dict(zip(k_list, vals))

    for deg in degree_sequence:
        k_freqNum_dict[deg] += 1
    # print(k_freq_dict)
    k_freq_dict = {k: v / n for k, v in k_freqNum_dict.items()}
    # print(k_freq_dict)

    plt.loglog(k_freq_dict.keys(), k_freq_dict.values(), 'b-', marker='o')
    plt.title("Degree Freqency Chart")
    plt.xlabel("Degree")
    plt.ylabel("Freqency")

    plt.savefig("G_Degree_Freqency.png")
    plt.show()
    # print(k_freq_dict)


    # 最小二乘拟合
    ##最小二乘法
    from scipy.optimize import leastsq  ##引入最小二乘法算法

    '''
         设置样本数据，真实数据需要在这里处理
    '''
    ##样本数据(Xi,Yi)，需要转换成数组(列表)形式
    Xi = np.array(list(k_freq_dict.keys()))
    Xi = np.log(Xi)
    Yi = np.array(list(k_freq_dict.values())) + 1e-10
    Yi = np.log(Yi)

    '''
        设定拟合函数和偏差函数
        函数的形状确定过程：
        1.先画样本图像
        2.根据样本图像大致形状确定函数形式(直线、抛物线、正弦余弦等)
    '''


    ##需要拟合的函数func :指定函数的形状
    def func(p, x):
        k, b = p
        return k * x + b


    ##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
    def error(p, x, y):
        return func(p, x) - y


    '''
        主要部分：附带部分说明
        1.leastsq函数的返回值tuple，第一个元素是求解结果，第二个是求解的代价值(个人理解)
        2.官网的原话（第二个值）：Value of the cost function at the solution
        3.实例：Para=>(array([ 0.61349535,  1.79409255]), 3)
        4.返回值元组中第一个值的数量跟需要求解的参数的数量一致
    '''

    # k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
    p0 = [1, 20]

    # 把error函数中除了p0以外的参数打包到args中(使用要求)
    Para = leastsq(error, p0, args=(Xi, Yi))

    # 读取结果
    k, b = Para[0]
    print("k=", k, "b=", b)
    print("cost：" + str(Para[1]))
    print("求解的拟合直线为:")
    print("y=" + str(round(k, 2)) + "x+" + str(round(b, 2)))

    '''
       绘图，看拟合效果.
       matplotlib默认不支持中文，label设置中文的话需要另行设置
       如果报错，改成英文就可以
    '''

    # 画样本点
    plt.figure(figsize=(8, 6))  ##指定图像比例： 8：6
    plt.scatter(Xi, Yi, color="green", label="Sample", linewidth=2)

    # 画拟合直线
    x = np.linspace(0, 12, 100)  ##在0-15直接画100个连续点
    y = k * x + b  ##函数式
    plt.plot(x, y, color="red", label="Line", linewidth=2)
    plt.legend(loc='lower right')  # 绘制图例
    plt.show()

    #
    degree_sequence_asc = sorted(k_freqNum_dict.values(), reverse=False)  # degree sequence
    dmax = max(degree_sequence_asc)
    import math

    print(dmax)
    bin_max = math.ceil(math.log(dmax, 2))
    print("bin_max: ", bin_max)
    print(len(k_freqNum_dict))

    bins = [0]
    for i in range(bin_max + 1):
        bins.append(pow(2, i))
    print(bins)

    # 对数分箱
    vc = pd.cut(degree_sequence_asc, bins).value_counts()
    # vc.index = vc.index.astype(str)
    x = []
    for intv in vc.index:
        x.append(intv.right)

    vc.index = x
    vc = dict(vc)
    print(vc)

    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.plot(vc.keys(), vc.values(), "o-")
    plt.show()

    # 最小二乘 对数分箱拟合
    ##最小二乘法 对数分箱拟合
    from scipy.optimize import leastsq  ##引入最小二乘法算法

    '''
         设置样本数据，真实数据需要在这里处理
    '''
    ##样本数据(Xi,Yi)，需要转换成数组(列表)形式
    Xi = np.array(list(k_freq_dict.keys()))
    Xi = np.log(Xi)
    Yi = np.array(list(k_freq_dict.values())) + 1e-10
    Yi = np.log(Yi)

    '''
        设定拟合函数和偏差函数
        函数的形状确定过程：
        1.先画样本图像
        2.根据样本图像大致形状确定函数形式(直线、抛物线、正弦余弦等)
    '''


    ##需要拟合的函数func :指定函数的形状
    def func(p, x):
        k, b = p
        return k * x + b


    ##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
    def error(p, x, y):
        return func(p, x) - y


    '''
        主要部分：附带部分说明
        1.leastsq函数的返回值tuple，第一个元素是求解结果，第二个是求解的代价值(个人理解)
        2.官网的原话（第二个值）：Value of the cost function at the solution
        3.实例：Para=>(array([ 0.61349535,  1.79409255]), 3)
        4.返回值元组中第一个值的数量跟需要求解的参数的数量一致
    '''

    # k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
    p0 = [1, 20]

    # 把error函数中除了p0以外的参数打包到args中(使用要求)
    Para = leastsq(error, p0, args=(Xi, Yi))

    # 读取结果
    k, b = Para[0]
    print("k=", k, "b=", b)
    print("cost：" + str(Para[1]))
    print("求解的拟合直线为:")
    print("y=" + str(round(k, 2)) + "x+" + str(round(b, 2)))

    '''
       绘图，看拟合效果.
       matplotlib默认不支持中文，label设置中文的话需要另行设置
       如果报错，改成英文就可以
    '''

    # 画样本点
    plt.figure(figsize=(8, 6))  ##指定图像比例： 8：6
    plt.scatter(Xi, Yi, color="green", label="Sample", linewidth=2)

    # 画拟合直线
    x = np.linspace(0, 12, 100)  ##在0-15直接画100个连续点
    y = k * x + b  ##函数式
    plt.plot(x, y, color="red", label="Line", linewidth=2)
    plt.legend(loc='lower right')  # 绘制图例
    plt.show()
    print('steps {steps}: gama = {gama}'.format(steps=len(vc.keys()), gama=k))

    from scipy import optimize


    def fit_line(x, a, b):
        return a * x + b


    Xi = np.array(list(k_freq_dict.keys()))
    x = np.log10(Xi)
    Yi = np.array(list(k_freq_dict.values())) + 1e-10
    y = np.log10(Yi)

    kmin = int(min(Xi))
    kmax = int(max(Xi))

    # 拟合
    a, b = optimize.curve_fit(fit_line, x, y)[0]
    print("斜率 k_ = ", a)
    print("a=-2.3, b=-0.2")

    x1 = np.arange(kmin, kmax, 0.01)
    y1 = (10 ** b) * (x1 ** a)

    plt.figure(figsize=(10, 6))
    plt.plot(Xi, Yi, 'go')
    plt.plot(x1, y1, 'b-')
    plt.xlabel("$k$")
    plt.ylabel("$p_k$")
    plt.ylim([1e-6, 1])
    plt.xscale("log")
    plt.yscale("log")

    import powerlaw

    data = [G.degree(i) for i in G.nodes()]
    print(max(data))

    fit = powerlaw.Fit(data)
    print(fit)
    kmin = fit.power_law.xmin
    print("kmin:", kmin)
    print("gamma:", fit.power_law.alpha)
    print("D:", fit.power_law.D)

    plt.figure(figsize=(6, 5))
    fig = fit.plot_pdf(marker='o', color='b', linewidth=1)
    fit.power_law.plot_pdf(color='b', linestyle='-', ax=fig)

    # 其他参数
    G = G_max_sub
    ac = nx.average_clustering(G)
    print('average_clustering: ', ac)

    G_dia = nx.diameter(G)
    G_avg_dist = nx.average_shortest_path_length(G)  # 所有节点间平均最短路径长度。
    print('diameter：', G_dia)
    print('average_shortest_path_length：', G_avg_dist)
    print(nx.assortativity.degree_assortativity_coefficient(G_max_sub))

    ass_coe = nx.degree_assortativity_coefficient(G)  # 度同配系数
    print("degree_assortativity_coefficient = ", ass_coe)

    dc = nx.degree_centrality(G)  # 邻点中心性 degree_v / (len(G) - 1), 度上限为len(G) - 1
    print('mean:', np.mean(list(dc.values())))
    print('degree_centrality = ', dc)

    cc = nx.closeness_centrality(G,
                                 distance='weight')  # 接近中心性 C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)}, s.t. n = len(G) - 1.0
    print('mean:', np.mean(list(cc.values())))
    print('closeness_centrality = ', cc)

    nbc = nx.betweenness_centrality(G, k=None, normalized=True)  # 点介数中心性：k=None表示全图，不限制跳数normalized=True表示用完全图边数作分母标准化
    print('mean:', np.mean(list(nbc.values())))
    print('betweenness_centrality = ', nbc)

    ebc = nx.edge_betweenness_centrality(G, k=None, normalized=True)  # 边介数中心性
    print('mean:', np.mean(list(ebc.values())))
    print('edge_betweenness_centrality = ', ebc)

    eigen_cent = nx.eigenvector_centrality_numpy(G,
                                                 weight='weight')  # 特征向量中心性（Eigenvector Centrality）。一个节点的重要性既取决于其邻居节点的数量（即该节点的度），也取决于其邻居节点的重要性。
    print('mean:', np.mean(list(eigen_cent.values())))
    print('eigenvector_centrality_numpy = ', eigen_cent)
