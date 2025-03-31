# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-31 10:31
#    @Description   : deepseek v3推断脚本
#
# ===============================================================


from openai import OpenAI
import os
from os import path as osp
from utils.time_util import readable_time_string
import json
import rouge
import argparse
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


class TestDataLLMInfer(object):
    def __init__(self, model_type, output_file_name, input_file_name):
        # 根据LLM模型区分API KEY和client
        self.mode_type = model_type
        if model_type == 'deepseek':
            self.api_key = os.getenv('DEEPSEEK_API_KEY')  # https://open.bigmodel.cn/dev/api/normal-model/glm-4
            self.client = OpenAI(api_key=self.api_key, base_url='https://api.deepseek.com')
            self.model_version = 'deepseek-chat'
        else:
            raise Exception('NOT TARGET MODEL_TYPE, PLEASE CHECK IT!!!')

        self.res = []
        self.output_file_name = output_file_name
        self.input_file_name = input_file_name
        with open('../data_prepare/user_prompt.txt', 'r', encoding='utf-8') as f_usr, \
            open('../data_prepare/sys_prompt.txt', 'r', encoding='utf-8') as f_sys:
            self.sys_prompt = f_sys.read()
            self.user_prompt = f_usr.read()

    def deepseek_generate(self, content, model_version):
        """基于Azure OPENAI CHATGPT的ABS生成"""
        user_prompt = self.user_prompt.replace('${CONTENT}', content)
        msg = [
            {'role': 'system', 'content': self.sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=model_version,
            messages=msg,
            # max_tokens=MAX_TOKENS,
            max_tokens=1000,
            temperature=0,
            top_p=0.000001,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
            response_format={
                'type': 'json_object'
            }
        )
        res_dict = json.loads(response.choices[0].message.content)
        # _, res_dict = json_repair_util.try_parse_json_object(res)

        return res_dict

    def pseudo_summary_generate_workflow(self):
        """
        基于LLM的abstractive summary生成
        Returns:
            None, 将结果存储到目标文件中
        """
        # 确定使用的模型
        if self.mode_type == 'deepseek':
            generate = self.deepseek_generate
        else:
            raise ValueError('MODEL TYPE IS WRONG!')

        # 获取数据集
        input_data_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.input_file_name)
        with open(input_data_path, mode='r', encoding='utf-8') as fr:
            text_lst = [line.strip() for line in fr.readlines()]
        print(f'Length of file lines: {len(text_lst)}')

        # 遍历文章, 获取pseudo summary, 每遍历一条就写一遍
        data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.output_file_name)
        with open(data_save_path, mode='a', encoding='utf-8') as fw:
            for i, line in enumerate(text_lst):
                if i < 19:
                    continue
                print(f'{readable_time_string("%Y%m%d %H:%M:%S")}: Processing No. {i+1}...')
                eles = line.split('\u0001')
                summary, content = eles[0], eles[1]
                content = content.replace('\n', '')
                res_dict = generate(content, self.model_version)
                print(f'-- {readable_time_string("%Y%m%d %H:%M:%S")}: {res_dict}')
                if res_dict:
                    fw.write('\u0001'.join([summary, content, res_dict['summary'].replace('\n', '')]) + '\n')
                else:
                    # print(f'LLM生成结果异常, 输入文本为: {content}')
                    pass

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
    parser.add_argument('--model_type', type=str, choices=['zhipu', 'openai', 'gemini', 'deepseek'], default='zhipu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # init object
    model_type = 'deepseek'
    input_file_name = 'companies_news_info_v2.test.txt'
    output_file_name = f'companies_news_info_v2.test.deepseek_infer.txt'

    inferor = TestDataLLMInfer(model_type, output_file_name, input_file_name)
    # inferor.pseudo_summary_generate_workflow()
    inferor._compute_rouges()
