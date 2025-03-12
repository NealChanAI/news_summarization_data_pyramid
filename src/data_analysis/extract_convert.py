# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-12 16:01
#    @Description   : 从摘要中找到与文本最相似的语句
#
# ===============================================================


import os
import json
import numpy as np
from tqdm import tqdm
import jieba
from rouge import Rouge
from os import path as osp
import warnings
warnings.filterwarnings("ignore")


maxlen = 256  # 初始化
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l"""
    if unit == 'word':
        source = jieba.cut(source, HMM=False)
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        rouge = Rouge()
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_metrics(source, target, unit='word'):
    """计算所有metrics"""
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics


def compute_main_metric(source, target, unit='word'):
    """计算主要metric"""
    return compute_metrics(source, target, unit)['main']


def text_segmentate(text, length=1, delimiters=u'\n。！？；：，'):
    """按照标点符号分割文本"""
    sentences = []
    buf = ''
    for ch in text:
        if ch in delimiters:
            if buf:
                sentences.append(buf)
            buf = ''
        else:
            buf += ch
    if buf:
        sentences.append(buf)
    return sentences


def text_split(text, limited=True):
    """将长句按照标点分割为多个子句"""
    texts = text_segmentate(text, 1, u'\n。；：，')
    if limited:
        texts = texts[:maxlen]
    return texts


def extract_matching(texts, summaries, start_i=0, start_j=0):
    """
    在texts中找若干句子，使得它们连起来与summaries尽可能相似
    最终找出文本和摘要中相似度较高的句子对，并将它们的索引返回
    """
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])  # 寻找摘要中最长的句子
    j = np.argmax([compute_main_metric(t, summaries[i], 'char') for t in texts])  # 寻找文本中与该摘要句子最相似的句子
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    return lm + [(start_i + i, start_j + j)] + rm


def extract_flow(inputs):
    """抽取式摘要的流程"""
    res = []
    for line in inputs:
        text, summary = line
        texts = text_split(text, True)
        summaries = text_split(summary, False)
        mapping = extract_matching(texts, summaries)
        labels = sorted(set([i[1] for i in mapping]))  # text的索引(已排序)
        pred_summary = ''.join([texts[i] for i in labels])
        metric = compute_main_metric(pred_summary, summary)
        res.append([texts, labels, summary, metric])
    return res


def load_data(filename):
    """加载数据"""
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            text = '\n'.join([d['sentence'] for d in l['text']])
            D.append((text, l['summary']))
    return D


def convert(data):
    """分句，并转换为抽取式摘要"""
    D = extract_flow(data)
    total_metric = sum([d[3] for d in D])
    D = [d[:3] for d in D]  # 排除metric指标, [texts, labels, summary]
    # print(u'抽取结果的平均指标: %s' % (total_metric / len(D)))
    return D


def _load_data(file_path):
    tar_file_path = osp.join(ROOT_DIR, 'data', 'THUCNews', file_path)
    res_data = []
    with open(tar_file_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            summary, text = line.strip().split('\u0001')
            res_data.append((text, summary))
    return res_data


if __name__ == '__main__':
    target_file_path = 'companies_news_info_v2.txt'
    data_lst = _load_data(target_file_path)
    res = convert(data_lst)
    print([i[1] for i in res])
