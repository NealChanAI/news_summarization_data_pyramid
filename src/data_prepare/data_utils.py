# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-18 14:15
#    @Description   : 数据集预处理函数集合
#
# ===============================================================


from get_lcsts_dataset import download_huggingface_data
import pandas as pd
import numpy as np
from os import path as osp
import sys
from utils.eval_util import compute_main_metric
import json
import pylcs
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def get_lcsts_data(file_path='lcsts_train.csv'):
    """获取LCSTS数据集"""
    def _iter_dataset(_file_path):
        """遍历文件, 存储到list中"""
        res = []
        with open(_file_path, mode='r', encoding='utf-8') as fr:
            for line in fr.readlines():
                summary, text = line.split('\u0001')
                res.append((summary, text))
        return res

    data_sava_path = osp.join(ROOT_DIR, 'data', file_path)
    if osp.exists(data_sava_path):
        res = _iter_dataset(data_sava_path)
    else:
        download_huggingface_data()
        res = _iter_dataset(data_sava_path)
    return res


def get_nlpcc_data(file_path='nlpcc_data.json'):
    """获取NLPCC数据集"""
    data_save_path = osp.join(ROOT_DIR, 'data', file_path)
    with open(data_save_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    res = []
    for line in dataset:
        res.append((line['title'], line['content']))

    return res


def get_shence_data(file_path='shence_data.json'):
    """获取shence数据集"""
    data_save_path = osp.join(ROOT_DIR, 'data', file_path)
    with open(data_save_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    res = []
    for line in dataset:
        res.append((line['title'], line['content']))

    return res


def get_thunews_data(file_path='sample_data_14000.csv'):
    """获取THUNews采样数据集"""
    data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', file_path)
    with open(data_save_path, 'r', encoding='utf-8') as f:
        res = [line for line in f.readlines()]
    return res


def get_500_top_companies_news_info(target_file, source_file='500强_news_info_raw.xlsx'):
    """获取业务方所需数据集"""
    data_path = osp.join(ROOT_DIR, 'data', 'THUCNews')
    source_file = osp.join(data_path, source_file)
    target_file = osp.join(data_path, target_file)

    df = pd.read_excel(source_file, engine='openpyxl')
    with open(target_file, mode='w', encoding='utf-8') as fw:
        for _, row in df.iterrows():
            abstract = row['abstract'].replace('\n', '')
            content = row['content'].replace('\n', '')
            fw.write('\u0001'.join([abstract, content]) + '\n')

