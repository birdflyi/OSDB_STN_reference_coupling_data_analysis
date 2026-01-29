#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2024/11/17 19:47
# @Author : 'Lou Zehua'
# @File   : Network_params_analysis.py
import os
import math
import traceback

import networkx as nx
import numpy as np
import pandas as pd
from GH_CoRE.working_flow import get_repo_name_fileformat, get_repo_year_filename, read_csvs
from matplotlib import pyplot as plt

from script.complex_network_analysis.build_network.build_Graph import DG2G
from script.complex_network_analysis.build_network.build_gh_collab_net import build_collab_net
from script.utils.timeout import timeout

plt.switch_backend('TkAgg')


def get_graph_feature(G, feat=None, timeout_sec=10*60):
    graph_feature_record = {
        "len_nodes": len(G.nodes),
        "len_edges": len(G.edges),
        "edge_density": 2 * len(G.edges) / (len(G.nodes) * (len(G.nodes) - 1)) if len(G.nodes) > 1 else 0,
        "is_sparse": None,
        "avg_deg": None,
        "avg_clustering": None,
        "lcc_node_coverage_ratio": None,
        "lcc_len_nodes": None,
        "lcc_len_edges": None,
        "lcc_edge_density": None,
        "lcc_diameter": None,
        "lcc_assort_coe": None,
        "lcc_avg_dist": None,
        "lcc_avg_deg_centr": None,
        "lcc_avg_close_centr": None,
        "lcc_avg_n_betw_centr": None,
        "lcc_avg_e_betw_centr": None,
        "lcc_avg_eigvec_centr": None,
    }
    if feat is None:
        feat = list(graph_feature_record.keys())

    # 宏观统计及zipf分布
    n = len(G.nodes())
    e = len(G.edges())
    e_threshold = n * math.log(n) if n > 0 else 0
    graph_feature_record["is_sparse"] = e < e_threshold

    avg_deg = 2 * e / n if n > 0 else 0
    graph_feature_record["avg_deg"] = avg_deg

    if "avg_clustering" in feat:
        avg_clust = nx.average_clustering(G)
        graph_feature_record['avg_clustering'] = avg_clust

    # higher complexity below
    if not any(["lcc_" in s for s in feat]):
        return {f: graph_feature_record[f] if f in graph_feature_record else None for f in feat}

    # largest connected components
    @timeout(seconds=timeout_sec)
    def connected_components(G, **kwargs):
        return nx.connected_components(G, **kwargs)

    try:
        if nx.is_connected(G):
            G_lcc = G.copy()
        else:
            lcc = max(connected_components(G), key=len)
            G_lcc = G.subgraph(lcc).copy()
        graph_feature_record["lcc_node_coverage_ratio"] = len(lcc) / len(G.nodes)
        graph_feature_record["lcc_len_nodes"] = len(G_lcc.nodes)
        graph_feature_record["lcc_len_edges"] = len(G_lcc.edges)
        graph_feature_record["lcc_edge_density"] = 2 * len(G_lcc.edges) / (len(G_lcc.nodes) * (len(G_lcc.nodes) - 1))
    except TimeoutError as e:
        print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))
        return graph_feature_record

    if 'lcc_diameter' in feat:
        @timeout(seconds=timeout_sec)
        def diameter(G, **kwargs):
            return nx.diameter(G, **kwargs)

        try:
            G_lcc_diameter = diameter(G_lcc)
            graph_feature_record['lcc_diameter'] = G_lcc_diameter
        except TimeoutError as e:
            print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    if 'lcc_assort_coe' in feat:
        @timeout(seconds=timeout_sec)
        def degree_assortativity_coefficient(G, **kwargs):
            return nx.degree_assortativity_coefficient(G, **kwargs)

        try:
            G_lcc_assort_coe = degree_assortativity_coefficient(G_lcc)  # 度同配系数
            graph_feature_record['lcc_assort_coe'] = G_lcc_assort_coe
        except TimeoutError as e:
            print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    if 'lcc_avg_dist' in feat:
        @timeout(seconds=timeout_sec)
        def average_shortest_path_length(G, **kwargs):
            return nx.average_shortest_path_length(G, **kwargs)

        try:
            G_lcc_avg_dist = average_shortest_path_length(G_lcc)  # 所有节点间平均最短路径长度。
            graph_feature_record['lcc_avg_dist'] = G_lcc_avg_dist
        except TimeoutError as e:
            print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    if 'lcc_avg_deg_centr' in feat:
        @timeout(seconds=timeout_sec)
        def degree_centrality(G, **kwargs):
            return nx.degree_centrality(G, **kwargs)

        try:
            G_lcc_deg_centr = degree_centrality(G_lcc)  # 度中心性 degree_v / (len(G) - 1), 度上限为len(G) - 1
            G_lcc_avg_deg_centr = np.mean(list(G_lcc_deg_centr.values()))
            graph_feature_record['lcc_avg_deg_centr'] = G_lcc_avg_deg_centr
        except TimeoutError as e:
            print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    if 'lcc_avg_close_centr' in feat:
        @timeout(seconds=timeout_sec)
        def closeness_centrality(G, **kwargs):
            return nx.closeness_centrality(G, **kwargs)

        try:
            G_lcc_close_centr = closeness_centrality(G_lcc, distance='weight')  # 接近中心性 C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)}, s.t. n = len(G) - 1.0
            G_lcc_avg_close_centr = np.mean(list(G_lcc_close_centr.values()))
            graph_feature_record['lcc_avg_close_centr'] = G_lcc_avg_close_centr
        except TimeoutError as e:
            print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    if 'lcc_avg_n_betw_centr' in feat:
        @timeout(seconds=timeout_sec)
        def betweenness_centrality(G, **kwargs):
            return nx.betweenness_centrality(G, **kwargs)

        try:
            G_lcc_n_betw_centr = betweenness_centrality(G_lcc, k=None,
                                                        normalized=True)  # 点介数中心性：k=None表示全图，不限制跳数normalized=True表示用完全图边数作分母标准化
            G_lcc_avg_n_betw_centr = np.mean(list(G_lcc_n_betw_centr.values()))
            graph_feature_record['lcc_avg_n_betw_centr'] = G_lcc_avg_n_betw_centr
        except TimeoutError as e:
            print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    if 'lcc_avg_e_betw_centr' in feat:
        @timeout(seconds=timeout_sec)
        def edge_betweenness_centrality(G, **kwargs):
            return nx.edge_betweenness_centrality(G, **kwargs)

        try:
            G_lcc_e_betw_centr = edge_betweenness_centrality(G_lcc, k=None, normalized=True)  # 边介数中心性
            G_lcc_avg_e_betw_centr = np.mean(list(G_lcc_e_betw_centr.values()))
            graph_feature_record['lcc_avg_e_betw_centr'] = G_lcc_avg_e_betw_centr
        except TimeoutError as e:
            print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    if 'lcc_avg_eigvec_centr' in feat:
        @timeout(seconds=timeout_sec)
        def eigenvector_centrality_numpy(G, **kwargs):
            return nx.eigenvector_centrality_numpy(G, **kwargs)

        try:
            G_lcc_eigvec_centr = eigenvector_centrality_numpy(G_lcc, weight='weight')  # 特征向量中心性（Eigenvector Centrality）。一个节点的重要性既取决于其邻居节点的数量（即该节点的度），也取决于其邻居节点的重要性。
            G_lcc_avg_eigvec_centr = np.mean(list(G_lcc_eigvec_centr.values()))
            graph_feature_record['lcc_avg_eigvec_centr'] = G_lcc_avg_eigvec_centr
        except TypeError:
            print("No available eigenvector_centrality!")
        except TimeoutError as e:
            print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    graph_feature_record = {f: graph_feature_record[f] if f in graph_feature_record else None for f in feat}
    return graph_feature_record


def plot_as_zipf_distribution(G, save_path=None):
    save_path = save_path or "./G_Degree_Rank_loglog_zipf.png"
    # zipf分布
    degree_dict = dict(nx.degree(G))
    degree_sequence = sorted(degree_dict.values(), reverse=True)  # degree sequence decrease order
    plt.loglog(np.array(range(len(degree_sequence))) + 1, degree_sequence, 'b-', marker='o')
    plt.title("Degree Rank Chart")
    plt.ylabel("Degree")
    plt.xlabel("Rank")

    plt.savefig(save_path)
    plt.show()


def plot_as_powerlaw_distribution(G, save_path=None):
    save_path = save_path or "./G_Degree_Freqency.png"
    # zipf分布
    degree_dict = dict(nx.degree(G))
    degree_sequence = sorted(degree_dict.values(), reverse=True)  # degree sequence decrease order

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

    plt.savefig(save_path)
    plt.show()
    # print(k_freq_dict)

    # -----------拟合-----------
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
        print("Warning: File format not found, returning empty graph.")
    return nx.MultiDiGraph()


if __name__ == '__main__':
    from etc import filePathConf

    repo_names = ["TuGraph-family/tugraph-db", "neo4j/neo4j", "facebook/rocksdb", "cockroachdb/cockroach"][0:2]
    year = 2023
    relation_extraction_save_dir = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                                              'repos_GH_CoRE')
    filenames_exists = os.listdir(relation_extraction_save_dir)
    if repo_names:
        repo_names_fileformat = list(map(get_repo_name_fileformat, repo_names))
        filenames = [get_repo_year_filename(s, year) for s in repo_names_fileformat]
        filenames = [filename for filename in filenames if filename in filenames_exists]
    else:
        filenames = filenames_exists

    df_dbms_repos_dict_tmp = read_csvs(relation_extraction_save_dir, filenames=filenames, index_col=None)
    df_dbms_repos_dict = {k: df_dbms_repos_dict_tmp[k] for k in sorted(df_dbms_repos_dict_tmp)}
    g_feat_path = os.path.join(filePathConf.absPathDict[filePathConf.GITHUB_OSDB_DATA_DIR],
                               'analysis_results/df_g_feat.csv')
    repo_keys = list(df_dbms_repos_dict.keys())
    if os.path.isfile(g_feat_path):
        df_g_feat_repo_key_as_index = pd.read_csv(g_feat_path, header="infer", index_col=None)
        repos_feat_dict_values = df_g_feat_repo_key_as_index.to_dict(orient="records")
        repos_feat_dict = {repos_feat_dict_values[k].get("repo_name", str(k)): repos_feat_dict_values[k]
                           for k in range(len(repos_feat_dict_values))}
    else:
        repos_feat_dict = {}

    for repo_key in repo_keys:
        if repo_key in repos_feat_dict.keys():
            continue

        graph_feature_record = {"repo_name": repo_key}
        df_dbms_repo = df_dbms_repos_dict[repo_key]
        G = build_collab_net(df_dbms_repo, src_tar_colnames=['src_entity_id', 'tar_entity_id'],
                             default_node_types=['src_entity_type', 'tar_entity_type'],
                             default_edge_type="event_type",
                             init_record_as_edge_attrs=True, use_df_col_as_default_type=True, out_g_type='G')
        graph_feature_record_complex_network = get_graph_feature(G)
        graph_feature_record.update(graph_feature_record_complex_network)
        repos_feat_dict[repo_key] = graph_feature_record
        df_g_feat = pd.DataFrame.from_dict(repos_feat_dict, orient='index')
        df_g_feat.to_csv(g_feat_path, index=False)
        print(f"{repo_key} saved into df_g_feat!")
