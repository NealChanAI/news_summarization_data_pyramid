# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-11 16:03
#    @Description   : LCSTS数据分析
#
# ===============================================================


import pandas as pd
from os import path as osp
import json


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
DATA_DIR = 'lcsts_data'


def parse_json(file_name):
    data_path = osp.join(ROOT_DIR, 'data', DATA_DIR, file_name)
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    res = []
    for rec in dataset:
        res.append((rec['content'], rec['title']))
    return res


def data_analysis(dataset):
    df = pd.DataFrame(dataset)
    df.columns = ['content', 'summary']
    print('length:', len(df))
    return df


def workflow(file_name):
    dataset = parse_json(file_name)
    df = data_analysis(dataset)

    df['content_length'] = df['content'].apply(len)  # 计算每行文本的字数
    df['summary_length'] = df['summary'].apply(len)  # 计算每行文本的字数

    content_length_mean = df['content_length'].mean()  # 均值
    content_length_max = df['content_length'].max()  # 最大值
    content_length_min = df['content_length'].min()  # 最小值

    summary_length_mean = df['summary_length'].mean()  # 均值
    summary_length_max = df['summary_length'].max()  # 最大值
    summary_length_min = df['summary_length'].min()  # 最小值

    print('content_length_mean:', content_length_mean)
    print('content_length_max:', content_length_max)
    print('content_length_min:', content_length_min)
    print('summary_length_mean:', summary_length_mean)
    print('summary_length_max:', summary_length_max)
    print('summary_length_min:', summary_length_min)


if __name__ == '__main__':
    file_name = 'lcsts_data.json'
    workflow(file_name)
