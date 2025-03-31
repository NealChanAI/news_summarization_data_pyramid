# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-21 10:22
#    @Description   : THUCNews Data Analysis
#
# ===============================================================


import pandas as pd
from os import path as osp
import os
import statistics


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
THUCNews_DIR = r'D:\neal\develop\datasets\THUCNews'


def check_length_distribution(file_path):
    """查询通用数据层的原文长度分布"""
    res_lst = []
    for dir in os.listdir(file_path):
        if dir == '财经':
            continue
        print(dir)
        dir = osp.join(file_path, dir)
        for i, f in enumerate(os.listdir(dir)):
            if i % 1000 == 0:
                print(i)
            f = osp.join(dir, f)
            with open(f, 'r', encoding='utf-8') as fr:
                res_lst.append(''.join(fr.readlines()))
    df = pd.DataFrame(res_lst)
    df.columns = ['text']
    df['text_length'] = df['text'].apply(lambda x: len(x))
    print(df['text_length'].describe())


def compute_mean_and_variance(file_path):
    file_path = osp.join(ROOT_DIR, 'data', 'THUCNews', file_path)
    with open(file_path, 'r', encoding='utf-8') as fr:
        data_lst = [line.strip() for line in fr.readlines()]
    text_lst = [len(i.split('\u0001')[1]) for i in data_lst]
    summary_lst = [len(i.split('\u0001')[0]) for i in data_lst]

    # test length analysis
    mean_value = statistics.mean(text_lst)
    stdev_value = statistics.stdev(text_lst)
    two_stdev = 2 * stdev_value
    lower_bound = mean_value - two_stdev
    upper_bound = mean_value + two_stdev
    print(mean_value, stdev_value, lower_bound, upper_bound)

    # summary length analysis
    mean_value = statistics.mean(summary_lst)
    stdev_value = statistics.stdev(summary_lst)
    two_stdev = 2 * stdev_value
    lower_bound = mean_value - two_stdev
    upper_bound = mean_value + two_stdev
    print(mean_value, stdev_value, lower_bound, upper_bound)


if __name__ == '__main__':
    # check_length_distribution(THUCNews_DIR)
    compute_mean_and_variance('companies_news_info_v2.txt')
