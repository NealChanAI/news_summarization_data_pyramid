# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-11 16:51
#    @Description   : 对XLSUM的数据进行格式转换, 使其符合模型要求
#
# ===============================================================


from datasets import load_dataset
from os import path as osp
import json


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def format_infer_dataset(data_type):
    """根据模型需求转换推断数据集格式"""
    data_path = osp.join(ROOT_DIR, 'data', 'chinese_simplified_XLSum_v2.0', f'chinese_simplified_{data_type}.jsonl')
    output_path = data_path[:-6] + '_formatted' + data_path[-6:]
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
    format_infer_dataset('test')
