# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-10 15:48
#    @Description   : ZHIPU chatGLM-4-Flash直接推断
#
# ===============================================================


import os
from os import path as osp
from zhipuai import core as zhipu_core
from zhipuai import ZhipuAI
import openai
from openai import AzureOpenAI
import sys
import addict
from utils.eval_util import compute_main_metric
from utils.token_utils import num_tokens_from_string
from utils.time_util import readable_time_string
from utils import json_repair_util
from utils.decorator_utils import retry_with_exponential_backoff
import json
import rouge
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
log = logging.getLogger(__name__)
ZHIPU_MODEL_VERSION = 'glm-4-flash'
# GEMINI_MODEL_VERSION = 'gemini-2.0-flash-thinking-exp-1219'
GEMINI_MODEL_VERSION = 'gemini-2.0-flash-exp'
OPENAI_MODEL_VERSION = 'gpt-4o'
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
        if model_type == 'zhipu':
            self.api_key = os.getenv('ZHIPU_API_KEY')  # https://open.bigmodel.cn/dev/api/normal-model/glm-4
            self.client = ZhipuAI(api_key=self.api_key)
            self.model_version = ZHIPU_MODEL_VERSION
        # elif model_type == 'gemini':
        #     self.api_key = os.getenv('GEMINI_API_KEY')
        #     self.client = genai.Client(  # https://aistudio.google.com/
        #             vertexai=False,
        #             api_key=self.api_key
        #         )
        #     self.model_version = GEMINI_MODEL_VERSION
        elif model_type == 'openai':  # openAI
            endpoint = os.getenv("AZURE_ENDPOINT_URL")
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")  # https://portal.azure.com/
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=self.api_key,
                # api_version="2024-05-01-preview",
                api_version="2024-08-01-preview",
            )
            self.model_version = OPENAI_MODEL_VERSION
        else:
            raise Exception('NOT TARGET MODEL_TYPE, PLEASE CHECK IT!!!')

        self.res = []
        self.output_file_name = output_file_name
        self.input_file_name = input_file_name
        with open('../data_prepare/user_prompt.txt', 'r', encoding='utf-8') as f_usr, \
            open('../data_prepare/sys_prompt.txt', 'r', encoding='utf-8') as f_sys:
            self.sys_prompt = f_sys.read()
            self.user_prompt = f_usr.read()

    @retry_with_exponential_backoff
    def zhipu_generate(self, content, model_version):
        """基于ZHIPU的ABS生成"""
        user_prompt = self.user_prompt.replace('${CONTENT}', content)
        msg = [
            {'role': 'system', 'content': self.sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=model_version,
                max_tokens=MAX_TOKENS,
                messages=msg,
                response_format={
                    'type': 'json_object'
                },
                temperature=0,
                top_p=0,
            )
        except zhipu_core._errors.APIRequestFailedError as e:
            print(e)
            return {}

        res = response.choices[0].message.content
        _, res_dict = json_repair_util.try_parse_json_object(res)

        # 若value不为string类型, 则返回空串
        if not isinstance(res_dict, dict):
            return {}
        if not isinstance(res_dict['summary'], str):
            if 'title' in res_dict['summary']:
                summary = res_dict['summary']['title']
                res_dict['summary'] = summary
            else:
                return
        return res_dict

    @retry_with_exponential_backoff
    def openai_generate(self, content, model_version):
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
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
            response_format={
                'type': 'json_object',
            }
        )

        res = response.choices[0].message.content
        _, res_dict = json_repair_util.try_parse_json_object(res)

        return res_dict

    # @retry_with_exponential_backoff
    # def gemini_generate(self, content, model_version):
    #     """基于Google Gemini的ABS生成"""
    #     user_prompt = self.user_prompt.replace('${CONTENT}', content)
    #     response = self.client.models.generate_content(
    #         model=model_version,
    #         config=types.GenerateContentConfig(
    #             system_instruction=self.sys_prompt,
    #             temperature=0,
    #             top_p=0,
    #             max_output_tokens=MAX_TOKENS if 'thinking' not in model_version else MAX_TOKENS * 2
    #             # response_mime_type='application/json',
    #             # response_schema={
    #             #     'required': ["text", "summary"],
    #             #     'properties': {
    #             #         "text": {"type": "string"},
    #             #         "summary": {"type": "integer"}
    #             #     },
    #             #     'type': 'OBJECT',
    #             # }
    #         ),
    #         contents=user_prompt
    #     )
    #
    #     res = response.text
    #     _, res_dict = json_repair_util.try_parse_json_object(response.text)
    #
    #     return res_dict

    def pseudo_summary_generate_workflow(self):
        """
        基于LLM的abstractive summary生成
        Returns:
            None, 将结果存储到目标文件中
        """
        # 确定使用的模型
        if self.mode_type == 'zhipu':
            generate = self.zhipu_generate
        # elif self.mode_type == 'gemini':
        #     generate = self.gemini_generate
        elif self.mode_type == 'openai':
            generate = self.openai_generate

        # 获取数据集
        input_data_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.input_file_name)
        with open(input_data_path, mode='r', encoding='utf-8') as fr:
            text_lst = [line.strip() for line in fr.readlines()]
        print(f'Length of file lines: {len(text_lst)}')

        # 遍历文章, 获取pseudo summary, 每遍历一条就写一遍
        data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.output_file_name)
        with open(data_save_path, mode='w', encoding='utf-8') as fw:
            for i, line in enumerate(text_lst):
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
    parser.add_argument('--model_type', type=str, choices=['zhipu', 'openai', 'gemini'], default='zhipu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # init object
    model_type = 'zhipu'
    input_file_name = 'companies_news_info_v2.test.txt'
    output_file_name = f'companies_news_info_v2.test.llm_infer.txt'

    inferor = TestDataLLMInfer(model_type, output_file_name, input_file_name)
    # inferor.pseudo_summary_generate_workflow()
    inferor._compute_rouges()
