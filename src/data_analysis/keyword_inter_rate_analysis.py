# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-20 10:59
#    @Description   : 
#
# ===============================================================


import numpy as np
from os import path as osp
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import rouge
import scipy.stats


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
HIT_STOPWORD_FILE = osp.join(ROOT_DIR, 'src', 'data_analysis', 'hit_stopwords.txt')
DATA_FILE = osp.join(ROOT_DIR, 'data', 'THUCNews', 'companies_news_info.txt')
MAX_LEN = 256


# def compute_rouge(source, target):
#     """计算rouge-1、rouge-2、rouge-l"""
#     source, target = ' '.join(source), ' '.join(target)
#     try:
#         scores = rouge.Rouge().get_scores(hyps=source, refs=target)
#         rouge_1 = scores[0]['rouge-1']['f']
#         rouge_2 = scores[0]['rouge-2']['f']
#         rouge_l = scores[0]['rouge-l']['f']
#         return 0.4 * rouge_1 + 0.4 * rouge_2 + 0.2 * rouge_l
#     except ValueError:
#         return 0.0

def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l"""
    if unit == 'word':
        source = jieba.cut(source, HMM=False)
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
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
        metrics['rouge-1'] * 1 + metrics['rouge-2'] * 0.0 +
        metrics['rouge-l'] * 0.0
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
        texts = texts[:MAX_LEN]
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


def compute_intersection_rate(lst_a, lst_b):
    """
    计算两个关键词列表的交叉率
    Args:
        lst_a: summary keywords list
        lst_b: text keywords list

    Returns: intersection rate

    """
    if not lst_a or not lst_b:
        return 0.0
    summary_keyword_set = set(lst_a)
    text_keyword_set = set(lst_b)
    intersection_count = len(text_keyword_set.intersection(summary_keyword_set))
    return intersection_count / len(lst_b)


def compute_keyword_apperance_rate(text_keywords, summary):
    """
    计算文本的关键词出现在summary中的概率
    Args:
        text_keywords:
        summary:

    Returns:

    """
    if not text_keywords or not summary:
        raise ValueError(f'{text_keywords}')

    cnt = 0
    for keyword in text_keywords:
        if keyword in summary:
            cnt += 1
    return cnt / len(text_keywords)


def extract_flow(inputs_file):
    """抽取式摘要的流程"""
    res = []
    stopwords = load_stopwords(HIT_STOPWORD_FILE)
    # for line in inputs:
    with open(inputs_file, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            summary, text = line.strip().split('\u0001')
            summary_keywords = tfidf_keyword_extraction_multiple_with_stopwords(summary, stopwords)
            text_keywords = tfidf_keyword_extraction_multiple_with_stopwords(text, stopwords)
            intersection_rate = compute_intersection_rate(summary_keywords, text_keywords)
            appearance_rate = compute_keyword_apperance_rate(text_keywords, summary)

            texts = text_split(text, True)
            summaries = text_split(summary, False)
            mapping = extract_matching(texts, summaries)
            labels = sorted(set([i[1] for i in mapping]))  # text的索引(已排序)
            pred_summary = ''.join([texts[i] for i in labels])
            metric = compute_main_metric(pred_summary, summary)
            res.append([texts, labels, summary, metric, intersection_rate, appearance_rate])
    return res


def load_stopwords(stopwords_file):
    """
    加载停用词列表.
    Args:
        stopwords_file (str): 停用词文件路径.

    Returns:
        set: 停用词集合.
    """
    stopwords = set()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())  # 去除行尾的换行符和空格
    return stopwords


def tfidf_keyword_extraction_multiple_with_stopwords(text, stopwords, top_k=30):
    """
    Extracts top keywords from multiple texts using TF-IDF.

    Args:
        texts (list): A list of input texts (strings).
        top_k (int): Number of top keywords to extract PER DOCUMENT.

    Returns:
        dict: A dictionary where keys are document indices and values are lists of top keywords for each document.
    """
    words = jieba.lcut(text)
    # 移除停用词
    filtered_words = [word for word in words if word not in stopwords]
    processed_text = ' '.join(filtered_words)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    feature_names = vectorizer.get_feature_names()
    tfidf_scores = tfidf_matrix.toarray().flatten()
    keywords_scores = list(zip(feature_names, tfidf_scores))
    keywords_scores = sorted(keywords_scores, key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, score in keywords_scores[:top_k]]

    return top_keywords


if __name__ == '__main__':
    res = extract_flow(DATA_FILE)
    for i in res:
        print(i[3], i[4])
    keyword_intersection_rate = [i[4] for i in res]
    appearance_rate = [i[5] for i in res]
    rouge_metric = [i[3] for i in res]
    # 计算 Pearson 相关系数和 p 值
    # correlation, p_value = scipy.stats.pearsonr(keyword_intersection_rate, rouge_metric)
    correlation, p_value = scipy.stats.pearsonr(appearance_rate, rouge_metric)

    print(f"Pearson 相关系数: {correlation:.4f}")
    print(f"P 值: {p_value:.4f}")
