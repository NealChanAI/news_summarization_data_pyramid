# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-12 20:19
#    @Description   : ChatGPT4推断
#
# ===============================================================


from os import path as osp
import rouge
import json
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
log = logging.getLogger(__name__)
TEST_DATA_PATH = osp.join(ROOT_DIR, 'data', 'THUCNews', 'companies_news_info_v2.test.txt')

'companies_news_info_v2.test.gpt4_infer.raw.txt'


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l"""
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


class ChatGPT4Inferor(object):
    def __init__(self, output_file_name, input_file_name):
        # 根据LLM模型区分API KEY和client
        self.res = []
        self.output_file_name = output_file_name
        self.input_file_name = input_file_name

    def chatgpt4_summary_generate_workflow(self):
        # 获取数据集
        input_data_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.input_file_name)
        raw_infer_file_name = self.output_file_name[:-4] + '.raw' + self.output_file_name[-4:]
        raw_infer_data_path = osp.join(ROOT_DIR, 'data', 'THUCNews', raw_infer_file_name)
        with open(input_data_path, mode='r', encoding='utf-8') as fr, \
            open(raw_infer_data_path, mode='r', encoding='utf-8') as fr_raw:
            text_lst = [line.strip() for line in fr.readlines()]
            raw_infer_lst = [line.strip() for line in fr_raw.readlines()]
        print(f'Length of file lines: {len(text_lst)}')

        # 遍历文章, 获取summary, 每遍历一条就写一遍
        data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.output_file_name)
        with open(data_save_path, mode='w', encoding='utf-8') as fw:
            for i, line in enumerate(text_lst):
                if i > 90:
                    continue
                eles = line.split('\u0001')
                summary, content = eles[0], eles[1]
                content = content.replace('\n', '')
                json_obj = json.loads(raw_infer_lst[i])
                pred_summary = json_obj['summary']
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


if __name__ == '__main__':
    input_file_name = 'companies_news_info_v2.test.txt'
    output_file_name = 'companies_news_info_v2.test.gpt4_infer.txt'

    inferor = ChatGPT4Inferor(output_file_name, input_file_name)
    inferor.chatgpt4_summary_generate_workflow()
    inferor._compute_rouges()



