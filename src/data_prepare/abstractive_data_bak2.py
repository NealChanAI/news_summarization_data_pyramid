# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-26 17:03
#    @Description   : 备份文件, Azure OpenAI chatGPT
#
# ===============================================================


import pandas as pd
import numpy as np
import os
from os import path as osp
from zhipuai import core as zhipu_core
from zhipuai import ZhipuAI
import openai
from google import genai
from google.genai import types
from openai import AzureOpenAI
import sys
import addict
from utils.eval_util import compute_main_metric
from utils.token_utils import num_tokens_from_string
from utils.time_util import readable_time_string
from utils import json_repair_util
from utils.decorator_utils import retry_with_exponential_backoff
from data_utils import get_thunews_data
import json
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


class PseudoSummaryAbstractive(object):
    def __init__(self, model_type, output_file_name, input_file_name, start_idx=0):
        # 根据LLM模型区分API KEY和client
        self.mode_type = model_type
        if model_type == 'zhipu':
            self.api_key = os.getenv('ZHIPU_API_KEY')  # https://open.bigmodel.cn/dev/api/normal-model/glm-4
            self.client = ZhipuAI(api_key=self.api_key)
            self.model_version = ZHIPU_MODEL_VERSION
        elif model_type == 'gemini':
            self.api_key = os.getenv('GEMINI_API_KEY')
            self.client = genai.Client(  # https://aistudio.google.com/
                    vertexai=False,
                    api_key=self.api_key
                )
            self.model_version = GEMINI_MODEL_VERSION
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
        self.start_idx = start_idx
        self.output_file_name = output_file_name
        self.input_file_name = input_file_name
        with open('user_prompt.txt', 'r', encoding='utf-8') as f_usr, \
            open('sys_prompt.txt', 'r', encoding='utf-8') as f_sys:
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
            print(e['error']['message'])
            return {}

        res = response.choices[0].message.content
        _, res_dict = json_repair_util.try_parse_json_object(res)

        return res_dict

    # @retry_with_exponential_backoff
    def openai_generate(self, content, model_version):
        """基于Azure OPENAI CHATGPT的ABS生成"""
        user_prompt = self.user_prompt.replace('${CONTENT}', content)
        print('num_tokens_from_string: ', num_tokens_from_string(user_prompt))
        msg = [
            {'role': 'system', 'content': self.sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=model_version,
            messages=msg,
            # max_tokens=MAX_TOKENS,
            max_tokens=800,
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

    @retry_with_exponential_backoff
    def gemini_generate(self, content, model_version):
        """基于Google Gemini的ABS生成"""
        user_prompt = self.user_prompt.replace('${CONTENT}', content)
        response = self.client.models.generate_content(
            model=model_version,
            config=types.GenerateContentConfig(
                system_instruction=self.sys_prompt,
                temperature=0,
                top_p=0,
                max_output_tokens=MAX_TOKENS if 'thinking' not in model_version else MAX_TOKENS * 2
                # response_mime_type='application/json',
                # response_schema={
                #     'required': ["text", "summary"],
                #     'properties': {
                #         "text": {"type": "string"},
                #         "summary": {"type": "integer"}
                #     },
                #     'type': 'OBJECT',
                # }
            ),
            contents=user_prompt
        )

        res = response.text
        _, res_dict = json_repair_util.try_parse_json_object(response.text)

        return res_dict

    def pseudo_summary_generate_workflow(self):
        """
        基于LLM的abstractive summary生成
        Returns:
            None, 将结果存储到目标文件中
        """
        # 确定使用的模型
        if self.mode_type == 'zhipu':
            generate = self.zhipu_generate
        elif self.mode_type == 'gemini':
            generate = self.gemini_generate
        elif self.mode_type == 'openai':
            generate = self.openai_generate

        # 获取数据集
        text_lst = get_thunews_data(self.input_file_name)

        # 遍历文章, 获取pseudo summary, 每遍历一条就写一遍
        data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.output_file_name)
        with open(data_save_path, mode='a', encoding='utf-8') as fw:
            for i, content in enumerate(text_lst):
                if i < self.start_idx:
                    continue
                print(f'{readable_time_string("%Y%m%d %H:%M:%S")}: Processing No. {i+1}...')
                res_dict = generate(content, self.model_version)
                print(f'-- {readable_time_string("%Y%m%d %H:%M:%S")}: {res_dict}')
                if res_dict:
                    fw.write('\u0001'.join([res_dict['summary'], content]))
                else:
                    print(f'LLM生成结果异常, 输入文本为: {content}')


def parse_args():
    """解析输入参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['zhipu', 'openai', 'gemini'], default='zhipu')
    args = parser.parse_args()
    return args


def _test():
    input_file_name = 'sample_data_10000_shizheng.csv'
    output_file_name = 'abstractive_pseudo_summary_datasets.csv'

    extractor_zhipu = PseudoSummaryAbstractive('zhipu', output_file_name, input_file_name)
    extractor_gemini = PseudoSummaryAbstractive('gemini', output_file_name, input_file_name)
    extractor_openai = PseudoSummaryAbstractive('openai', output_file_name, input_file_name)

    text_lst = get_thunews_data(input_file_name)
    content = text_lst[0]
    print(num_tokens_from_string(content))
    # print('智谱:')
    # print(extractor_zhipu.zhipu_generate(content, ZHIPU_MODEL_VERSION))
    # print('-' * 20)
    # print('GOOGLE:')
    # print(extractor_gemini.gemini_generate(content, GEMINI_MODEL_VERSION))
    # print('-' * 20)
    print('OPENAI:')
    print(extractor_openai.openai_generate(content, OPENAI_MODEL_VERSION))
    # print('-' * 10)
    # content = text_lst[2]
    # print(num_tokens_from_string(content))
    # print(extractor_openai.openai_generate(content, OPENAI_MODEL_VERSION))


def _test2():
    input_file_name = 'sample_data_10000_shizheng.csv'
    output_file_name = 'abstractive_pseudo_summary_datasets.csv'

    extractor_gemini = PseudoSummaryAbstractive('gemini', output_file_name, input_file_name)

    text_lst = get_thunews_data(input_file_name)
    content = text_lst[8]
    extractor_gemini.gemini_generate(content, GEMINI_MODEL_VERSION)


if __name__ == '__main__':
    # init object
    mode_type = 'openai'
    input_file_name = 'sample_data_10000_shizheng.csv'
    output_file_name = f'abstractive_pseudo_summary_datasets_{mode_type}.csv'
    start_idx = 0

    extractor = PseudoSummaryAbstractive(mode_type, output_file_name, input_file_name, start_idx)
    # extractor.pseudo_summary_generate_workflow()

    _test()
    # _test2()
