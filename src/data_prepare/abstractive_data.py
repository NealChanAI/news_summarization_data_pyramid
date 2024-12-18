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
import sys
import addict
from utils.eval_util import compute_main_metric
from data_utils import get_thunews_data
import json
import argparse
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


def parse_args():
    """解析输入参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='chatGLM-4')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pass
