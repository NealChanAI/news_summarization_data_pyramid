# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-18 16:48
#    @Description   : 基于LLM的abstractive summary生成
#
# ===============================================================


import pandas as pd
import numpy as np
from os import path as osp
from zhipuai import ZhipuAI
import openai
import sys
import addict
from utils.eval_util import compute_main_metric
from data_utils import get_thunews_data
import json
import argparse
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
API_KEY = '9fa620ace8671f241eccad128bfad0bd.wtQvQFooUFGB7TWi'  # 智谱API key


class PseudoSummaryAbstractive(object):
    def __init__(self, model_type, output_file_name, input_file_name):
        # 根据LLM模型区分API KEY和client
        self.mode_type = model_type
        if model_type == 'zhipu':
            self.api_key = os.getenv('ZHIPU_API_KEY')
            self.client = ZhipuAI(api_key=zhipu_api_key)
        else:
            self.api_key = os.getenv('OPENAI_API_KEY')
            self.client = openai.OpenAI()

        client = openai.OpenAI()
        self.res = []
        self.output_file_name = output_file_name
        self.input_file_name = input_file_name
        self.sys_prompt = '你是一个资深的新闻摘要撰写专家，擅长为华语新闻撰写高质量的摘要'
        self.user_prompt = '''
            # 角色：你是一名华语新闻摘要撰写专家
            # 场景：根据提供的新闻内容，撰写一份高质量的中文摘要
            # 需要分析的【新闻内容】：${CONTENT}
            
            # 任务：
            1. 理解与分析提供给你的【新闻内容】；
            2. 根据【内容】撰写一篇高质量的【新闻摘要】；
            
            # 【字段定义】
            请严格按照以key为【新闻内容】，value为【新闻摘要】的格式输出JSON字符串，不要输出【json】【\n】字符串，不要输出python代码，不要返回多余信息:
            
            # 注意事项：
            1. 必须严格按照【字段定义】中的格式输出；
            2. 必须严格避免出现政治敏感、色情、赌博相关的词汇；
        '''


    def zhipu_generate(self, content):
        """基于ZHIPU的ABS生成"""
        user_prompt = self.user_prompt.replace('${CONTENT}', content)
        msg = [
            {'role': 'system', 'content': self.sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=MODEL_TYPE,
            messages=msg,
            temperature=0,
            top_p=0,
        )

        return response.choices[0].message.content


    def openai_generate(self, content):
        """基于OPENAI CHATGPT的ABS生成"""
        pass


    def _is_valid(self, llm_res):
        """验证大模型生成的数据格式是否符合要求"""
        if not llm_res:
            return False, 'is_none'
        try:
            llm_res_dict = json.loads(llm_res)
        except BaseException as e:
            return False, 'not_json_format'
        return True, ''


    def _llm_res_format(self, res_str):
        """对llm生成的结果作格式化预处理"""
        res_str_format = res_str.replace('json', '').replace("'", '').replace('\n', '') \
                .replace('`', '').replace(' ', '')
        return res_str_format


    def parse_llm_res(self, llm_res):
        pass


    def pseudo_summary_generate_workflow(self):
        """
        基于LLM的abstractive summary生成
        Returns:
            None, 将结果存储到目标文件中
        """
        # 获取数据集
        text_lst = get_thunews_data(self.input_file_name)
        # 遍历文章, 获取伪摘要
        generate = zhipu_generate if self.mode_type == 'zhipu' else openai_generate
        for content in text_lst:
            llm_res = generate(content)
            flag, err_msg = self._is_valid(llm_res)
            if flag:
                self.res.append('\u0001'.join([llm_res, content]))

        df = pd.DataFrame(self.res)
        data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.output_file_name)
        df.to_csv(data_save_path, index=False)



def parse_args():
    """解析输入参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='chatGLM-4')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # init object
    output_file_name = 'abstractive_pseudo_summary_datasets.csv'

