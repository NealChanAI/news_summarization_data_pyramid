# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-01-07 15:14
#    @Description   : 
#
# ===============================================================


from data_utils import get_500_top_companies_news_info
from os import path as osp


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def get_target_dataset():
    target_file = 'companies_news_info.txt'
    get_500_top_companies_news_info(target_file)


if __name__ == '__main__':
    get_target_dataset()
