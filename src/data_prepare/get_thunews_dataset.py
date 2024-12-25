# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-18 10:27
#    @Description   : THUNews数据集预处理
#
# ===============================================================


import pandas as pd
from os import path as osp
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
SAMPLE_NUMS = 10000


def data_sample(dir_path, sample_nums=10000):
    """
    对指定类别的新闻数据集进行采样
    Args:
        dir_path: 特定类型的数据集目录

    Returns: 指定类别的采样新闻数据集

    """
    tar_path = osp.join(ROOT_DIR, 'data', dir_path)
    files = os.listdir(tar_path)

    np.random.seed(1024)  # 设置随机种子
    np.random.shuffle(files)  # 随机打乱数据
    sampled_files = files[:sample_nums]

    res = []
    for file in sampled_files:
        tar_file_path = osp.join(tar_path, file)
        with open(tar_file_path, 'r', encoding='utf-8') as fr:
            content = fr.read()
            content = content.replace('\n', '。').replace('\r\n', '。')
        res.append(content)

    return res


def workflow(result_path, sample_nums, dataset_path='THUCNews'):
    """
    完整工作流
    Args:
        sample_nums: 每个类别的采样数
        dataset_path: THUNews数据集目录
    Returns:
    """
    res = []
    dir_path = osp.join(ROOT_DIR, 'data', dataset_path)
    cata_lst = os.listdir(dir_path)
    for cata in cata_lst:
        print('类别:', cata)
        if cata != '时政':
            continue
        cata_res = data_sample(osp.join(dataset_path, cata), sample_nums)
        res.extend(cata_res)
    df = pd.DataFrame(res)
    df.to_csv(osp.join(ROOT_DIR, 'data', dataset_path, result_path), index=False, header=False)


if __name__ == '__main__':
    # result_path = 'sample_data_14000.csv'
    # result_path = 'sample_data_1400.csv'
    result_path = f'sample_data_{SAMPLE_NUMS}_shizheng.csv'
    workflow(result_path, SAMPLE_NUMS)
