# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-01-07 15:14
#    @Description   : 500强info数据集预处理
#
# ===============================================================


from data_utils import get_500_top_companies_news_info_v2
from data_utils import train_test_split
from os import path as osp


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def get_target_dataset():
    target_file = 'companies_news_info_v2.txt'
    get_500_top_companies_news_info_v2(target_file)

def dataset_train_test_split():
    target_file = 'companies_news_info_v2.txt'
    ratio = 0.6
    train_test_split(target_file, ratio)


if __name__ == '__main__':
    get_target_dataset()
    dataset_train_test_split()
