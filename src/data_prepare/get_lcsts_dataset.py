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


def download_huggingface_data(data_type):
    dataset = load_dataset("hugcyp/LCSTS", split=data_type)  # 仅获取训练数据集
    df = dataset.to_pandas()
    data_type = data_type.replace('validation', 'val')
    data_save_path = osp.join(ROOT_DIR, 'data', 'lcsts_data',f'lcsts_{data_type}.csv')
    df.to_csv(data_save_path, index=False, sep='\u0001')


def format_infer_dataset(data_type):
    """根据模型需求转换推断数据集格式"""
    data_type = data_type.replace('validation', 'val')
    data_path = osp.join(ROOT_DIR, 'data', 'lcsts_data', f'lcsts_{data_type}.csv')
    output_path = data_path[:-4] + '_formatted' + data_path[-4:]
    res = []
    with open(data_path, 'r', encoding='utf-8') as fr, open(output_path, 'w', encoding='utf-8') as fw:
        for i, line in enumerate(fr.readlines()):
            if i == 0:
                continue
            summary, text = line.strip().split('\u0001')
            res.append('\u0001'.join([summary, text]))
        fw.writelines('\n'.join(res))


if __name__ == '__main__':
    download_huggingface_data('train')
    download_huggingface_data('validation')
    download_huggingface_data('test')
    format_infer_dataset('validation')

