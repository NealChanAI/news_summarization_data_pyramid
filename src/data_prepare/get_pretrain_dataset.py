# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-01-10 14:49
#    @Description   : 对Zhipu数据集进行拆分, 拆分为训练集和测试集
#
# ===============================================================


from data_utils import get_500_top_companies_news_info_v2
from data_utils import train_test_split
from os import path as osp


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def dataset_train_test_split():
    target_file = 'abstractive_pseudo_summary_datasets_zhipu.csv'
    ratio = 0.9
    train_test_split(target_file, ratio)


if __name__ == '__main__':
    dataset_train_test_split()