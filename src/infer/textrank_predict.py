# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-12 17:35
#    @Description   : 基于TextRankZH的推断
#
# ===============================================================


from os import path as osp
import rouge
import argparse
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from utils.time_util import readable_time_string
import logging
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
log = logging.getLogger(__name__)
MAX_TOKENS = 1024
TEST_DATA_PATH = osp.join(ROOT_DIR, 'data', 'THUCNews', 'companies_news_info_v2.test.txt')


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l"""
    # print(f'source: {source}')
    # print(f'target: {target}')
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


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


class TextrankInferor(object):
    def __init__(self, sentence_nums, output_file_name, input_file_name):
        # 根据LLM模型区分API KEY和client
        self.res = []
        self.sentence_nums = sentence_nums
        self.output_file_name = output_file_name
        self.input_file_name = input_file_name
        self.tr4s = TextRank4Sentence()

    def textrank_summary_generate_workflow(self):
        """
        基于TextRank的extractive summary生成
        Returns:
            None, 将结果存储到目标文件中
        """
        # 获取数据集
        input_data_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.input_file_name)
        with open(input_data_path, mode='r', encoding='utf-8') as fr:
            text_lst = [line.strip() for line in fr.readlines()]
        print(f'Length of file lines: {len(text_lst)}')

        # 遍历文章, 获取pseudo summary, 每遍历一条就写一遍
        data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.output_file_name)
        with open(data_save_path, mode='w', encoding='utf-8') as fw:
            for i, line in enumerate(text_lst):
                eles = line.split('\u0001')
                summary, content = eles[0], eles[1]
                content = content.replace('\n', '')
                self.tr4s.analyze(text=content, lower=True, source='all_filters')
                pred_summary_lst = self.tr4s.get_key_sentences(num=self.sentence_nums)
                pred_summary = '。'.join([i.sentence for i in pred_summary_lst])
                fw.write('\u0001'.join([summary, content, pred_summary.replace('\n', '')]) + '\n')

    def _compute_rouges(self):
        """计算Rouge指标"""
        gens, summaries = [], []
        data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.output_file_name)
        with open(data_save_path, mode='r', encoding='utf-8') as fr:
            for line in fr.readlines():
                line = line.strip()
                summary, content, gen_summary = line.split('\u0001')
                gens.append(gen_summary)
                summaries.append(summary)
        scores = compute_rouges(gens, summaries)
        print(scores)


def parse_args():
    """解析输入参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['zhipu', 'openai', 'gemini'], default='zhipu')
    args = parser.parse_args()
    return args


def test():
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    text = lines[0]
    print(text)
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')

    print('摘要：')
    print(type(tr4s.get_key_sentences(num=3)))
    print(type(tr4s.get_key_sentences(num=3)[0]))

    for item in tr4s.get_key_sentences(num=3):
        print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重


if __name__ == '__main__':
    input_file_name = 'companies_news_info_v2.test.txt'
    output_file_name = 'companies_news_info_v2.test.textrank_infer.txt'

    inferor = TextrankInferor(4, output_file_name, input_file_name)
    inferor.textrank_summary_generate_workflow()
    inferor._compute_rouges()
    # test()
