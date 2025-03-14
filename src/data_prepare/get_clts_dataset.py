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
import random


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


def dataset_sample(file_path, num_sample):
    """从数据集集中随机采样"""
    data_path = osp.join(ROOT_DIR, 'data', 'clts_data', file_path)
    output_train_path = data_path[:-4] + '_clts_data' + f'.{num_sample}' + data_path[-4:]

    # 读取所有行
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]

    # 随机采样
    random.seed(1024)
    sampled_lines = random.sample(lines, num_sample)

    with open(output_train_path, 'w', encoding='utf-8') as f_train:
        f_train.writelines('\n'.join(sampled_lines))


def split_train_and_valid(file_path):
    """拆分数据集"""
    file_path = osp.join(ROOT_DIR, 'data', 'clts_data', file_path)
    with open(file_path, 'r', encoding='utf-8') as fr:
        lst = [line.strip() for line in fr.readlines()]
    train_file_path = file_path.replace('10000', '9000')
    val_file_path = file_path.replace('10000', '1000')
    with open(train_file_path, 'w', encoding='utf-8') as f_train, open(val_file_path, 'w', encoding='utf-8') as f_val:
        f_train.writelines('\n'.join(lst[:9000]))
        f_val.writelines('\n'.join(lst[9000:]))


if __name__ == '__main__':
    # format_infer_dataset('test.src', 'test.tgt')
    # format_infer_dataset('train.src', 'train.tgt')
    # format_infer_dataset('valid.src', 'valid.tgt')
    # dataset_sample('train_formatted.txt', 10000)
    split_train_and_valid('train_formatted_clts_data.10000.txt')
