#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2026/2/4 18:51
# @Author : 'Lou Zehua'
# @File   : df_mem_opt.py

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

import pandas as pd

from typing import Dict, List, Union


def merge_dict_dfs_memory_efficient(df_dict: Dict[str, pd.DataFrame],
                                    k_col: str = 'k',
                                    v_cols: Union[List[str], None] = None,
                                    batch_size: int = 50) -> pd.DataFrame:
    """
    内存优化的字典DataFrame合并去重函数
    适用于大规模字典和大DataFrame场景
    """
    if not df_dict:
        target_cols = [k_col] + (v_cols or [])
        return pd.DataFrame(columns=target_cols)

    # 自动推断v_cols（如果不提供）
    if v_cols is None:
        sample_df = next(iter(df_dict.values()))
        v_cols = [col for col in sample_df.columns if col != k_col]

    # 构建目标列名列表
    target_cols = [k_col] + v_cols

    # 使用生成器逐个处理DataFrame，节省内存
    def extract_columns_generator():
        processed_count = 0
        for fname, df in df_dict.items():
            try:
                # 仅提取目标列，避免复制整个DataFrame
                subset = df[target_cols].copy()
                processed_count += 1
                yield subset
            except KeyError as e:
                missing_cols = set(target_cols) - set(df.columns)
                raise ValueError(f"DataFrame '{fname}'缺少必要列: {missing_cols}") from e

    # 分批处理避免内存峰值
    chunks = []
    temp_combined = []

    for i, df_subset in enumerate(extract_columns_generator()):
        temp_combined.append(df_subset)

        # 达到批次大小时进行合并和初步去重
        if (i + 1) % batch_size == 0 and temp_combined:
            if len(temp_combined) > 1:
                batch_df = pd.concat(temp_combined, ignore_index=True)
            else:
                batch_df = temp_combined[0]

            # 批次内部去重
            batch_df = batch_df.drop_duplicates(subset=[k_col], keep='first')
            chunks.append(batch_df)
            temp_combined = []  # 清空临时存储

            # 当累积足够多的批次后再做一次全局去重(进一步优化内存)
            if len(chunks) >= 3:  # 控制中间结果的数量
                intermediate_result = pd.concat(chunks, ignore_index=True)
                intermediate_result = intermediate_result.drop_duplicates(
                    subset=[k_col], keep='first'
                )
                chunks = [intermediate_result]  # 替换为压缩后的结果

    # 处理剩余未满一批的数据
    if temp_combined:
        if len(temp_combined) > 1:
            remaining_df = pd.concat(temp_combined, ignore_index=True)
        else:
            remaining_df = temp_combined[0]
        remaining_df = remaining_df.drop_duplicates(subset=[k_col], keep='first')
        chunks.append(remaining_df)

    # 最终合并所有批次并去重
    if not chunks:
        return pd.DataFrame(columns=target_cols)
    elif len(chunks) == 1:
        final_df = chunks[0]
    else:
        final_df = pd.concat(chunks, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=[k_col], keep='first')

    # 重新整理索引
    final_df = final_df.reset_index(drop=True)[target_cols]  # 确保列顺序正确

    return final_df


if __name__ == "__main__":
    # 示例演示及测试
    # 创建大型示例数据来展示内存效率
    large_df_dict = {}

    # 生成多个DataFrame模拟真实大数据集
    for i in range(10):  # 10个不同的DataFrame
        data_length = 5000  # 每个DF有5000行
        df_large = pd.DataFrame({
            'k': range(i * data_length, (i + 1) * data_length),
            'v1': [f'value_{j}_part{i}' for j in range(data_length)],
            'v2': [(j + i * 1000) % 100 for j in range(data_length)],
            'extra_col_A': ['A'] * data_length,
            'extra_col_B': ['B'] * data_length,
            'extra_col_C': list(range(data_length))
        })
        large_df_dict[f'dataframe_{i}'] = df_large

    # 引入一些重复key以检验去重效果
    overlap_df = pd.DataFrame({
        'k': [100, 200, 300],  # 这些key可能已在前面出现过
        'v1': ['overlap_val1', 'overlap_val2', 'overlap_val3'],
        'v2': [999, 888, 777],
        'extra_col_A': ['X'] * 3,
        'extra_col_B': ['Y'] * 3,
        'extra_col_C': [1, 2, 3]
    })
    large_df_dict['overlap_data'] = overlap_df

    print("开始执行内存优化合并...")
    print(f"输入字典包含 {len(large_df_dict)} 个DataFrame")

    # 执行合并操作
    result = merge_dict_dfs_memory_efficient(
        large_df_dict,
        k_col='k',
        v_cols=['v1', 'v2'],
        batch_size=3  # 设置较小的batch size以便观察分批效果
    )

    print("\n=== 合并结果 ===")
    print(f"合并后共有 {len(result)} 行记录")
    print("前10条记录:")
    print(result.head(10))
    print("\n后10条记录:")
    print(result.tail(10))

    # 验证去重效果
    duplicate_check = result.duplicated(subset=['k']).sum()
    print(f"\n检查重复项: 发现 {duplicate_check} 个重复的k值")

    # 显示内存相关信息
    memory_usage_mb = result.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"结果DataFrame内存占用: {memory_usage_mb:.2f} MB")
