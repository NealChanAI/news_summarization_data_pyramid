# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-17 9:58
#    @Description   : 获取CLTS数据集
#
# ===============================================================


from datasets import load_dataset
from os import path as osp


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def _format_infer_dataset(data_type):
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


def format_infer_dataset(source_file, target_file):
    """根据模型需求转换推断数据集格式"""
    source_file = osp.join(ROOT_DIR, 'data', 'clts_data', source_file)
    target_file = osp.join(ROOT_DIR, 'data', 'clts_data', target_file)
    output_file = source_file[:-4] + '_formatted.txt'
    with open(source_file, mode='r', encoding='utf-8') as f_src, \
        open(target_file, mode='r', encoding='utf-8') as f_tgt, \
        open(output_file, mode='w', encoding='utf-8') as f_opt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

        if len(src_lines) != len(tgt_lines):
            raise ValueError('长度不一致')

        res = []
        for i in range(len(src_lines)):
            text = src_lines[i].strip().replace(' ', '')
            summary = tgt_lines[i].strip().replace(' ', '')
            res.append('\u0001'.join([summary, text]))
        f_opt.writelines('\n'.join(res))


if __name__ == '__main__':
    source = 'test.src'
    target = 'test.tgt'
    format_infer_dataset(source, target)
