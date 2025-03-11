# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-11 16:51
#    @Description   : 从HuggingFace中获取CNewSum数据集, 并保存到本地
#
# ===============================================================


from datasets import load_dataset
from os import path as osp
import json


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
HUGGINGFACE_DATASET_PATH = 'ethanhao2077/cnewsum-processed'


def tranform_and_save_data(data_type='train'):
    """从huggingface中加载与保存数据集"""
    dataset = load_dataset(HUGGINGFACE_DATASET_PATH, split=data_type)
    data_type = 'val' if data_type == 'dev' else data_type
    res = []
    for line in dataset:
        # dict type
        tmp_res = {'text': line['article'], 'summary': line['summary']}
        json_string = json.dumps(tmp_res, ensure_ascii=False)
        res.append(json_string)

    with open(osp.join(ROOT_DIR, 'data', 'CNewSum_v2', f'{data_type}_dataset.txt'), mode='w', encoding='utf-8') as fw:
        fw.writelines('\n'.join(res))


def format_infer_dataset(data_type):
    """根据模型需求转换推断数据集格式"""
    data_type = data_type.replace('dev', 'val')
    data_path = osp.join(ROOT_DIR, 'data', 'CNewSum_v2', f'{data_type}_dataset.txt')
    output_path = data_path[:-4] + '_formatted' + data_path[-4:]
    res = []
    with open(data_path, 'r', encoding='utf-8') as fr, open(output_path, 'w', encoding='utf-8') as fw:
        for line in fr.readlines():
            line = line.strip()
            json_obj = json.loads(line)
            summary = json_obj['summary']
            text = json_obj['text']
            res.append('\u0001'.join([summary, text]))
        fw.writelines('\n'.join(res))


if __name__ == '__main__':
    tranform_and_save_data('train')
    tranform_and_save_data('dev')
    tranform_and_save_data('test')
    format_infer_dataset('dev')
