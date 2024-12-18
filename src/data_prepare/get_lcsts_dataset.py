# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-17 9:58
#    @Description   : 获取LCSTS数据集
#
# ===============================================================


from datasets import load_dataset
from os import path as osp


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def download_huggingface_data():
    dataset = load_dataset("hugcyp/LCSTS", split="train")  # 仅获取训练数据集
    df = dataset.to_pandas()

    data_save_path = osp.join(ROOT_DIR, 'data', 'lcsts_train.csv')
    df.to_csv(data_save_path, index=False, sep='\u0001')


if __name__ == '__main__':
    download_huggingface_data()
