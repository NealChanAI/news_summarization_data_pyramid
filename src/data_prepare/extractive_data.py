# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-17 10:55
#    @Description   : generate extractive summarization training data using GSG
#
# ===============================================================


import pandas as pd
import numpy as np
from os import path as osp
import sys
from utils.eval_util import compute_main_metric
from data_utils import get_thunews_data
import json
import pylcs
import warnings
warnings.filterwarnings("ignore")


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录


class PseudoSummaryExtractive(object):
    def __init__(self, output_file_name, input_file_name='sample_data_14000.csv'):
        self.res = []
        self.output_file_name = output_file_name
        self.input_file_name = input_file_name

    def principal_GSG(self, article, topk=3):
        """
        PEGASUS GSG算法: Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. 2020. Pegasus: Pre-training with extracted  gap-sentences for abstractive summarization. In International Conference on Machine Learning, pages  11328–11339. PMLR.
        Args:
            article: news content
            topk: 摘要的句数上限

        Returns: 摘要内容

        """
        sentences = article.split(".")  # 全部句子
        # 每个句子大于6个单词才认为是一个有效句子
        clean_sentences = [sen + "." for sen in sentences if len(sen.split(" ")) > 6]
        principal_sens = []  # 核心句子
        for _ in range(topk):
            scores_f1 = []
            for cur_sentence in clean_sentences:
                left_sentences = []
                for sen in sentences:
                    if sen not in principal_sens and sen != cur_sentence:
                        left_sentences.append(sen)
                left_article = "".join(left_sentences)
                selected_sentences = "".join(principal_sens + [cur_sentence])
                cur_scores = scorer.score(selected_sentences, left_article)
                scores_f1.append(cur_scores["rouge1"][2])
            if len(scores_f1) == 0: continue
            max_idx = np.argmax(scores_f1)
            principal_sens.append(clean_sentences[max_idx])
            clean_sentences.pop(max_idx)
        return principal_sens


    def _text_segmentate(self, text, length=1, delimiters=u'\n。；：，'):
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

    def text_process(self, text):
        """分割文本"""
        texts = _text_segmentate(text, 32, u'\n。')
        result, length = [], 0
        for text in texts:
            if length + len(text) > maxlen * 1.5 and len(result) >= 3:
                yield result
                result, length = [], 0
            result.append(text)
            length += len(text)
        if result and len(result) >= 3:
            yield result


    def _gather_join(self, texts, idxs):
        """取出对应的text，然后拼接起来"""
        return ''.join([texts[i] for i in idxs])


    def pseudo_summary(self, text):
        """构建伪标签摘要数据集"""
        source_idxs, target_idxs = list(range(len(text))), []
        while True:
            sims = []
            for i in source_idxs:
                new_source_idxs = [j for j in source_idxs if j != i]
                new_target_idxs = sorted(target_idxs + [i])
                new_source = gather_join(text, new_source_idxs)
                new_target = gather_join(text, new_target_idxs)
                sim = pylcs.lcs(new_source, new_target)
                sims.append(sim)
            new_idx = source_idxs[np.argmax(sims)]
            source_idxs.remove(new_idx)
            target_idxs = sorted(target_idxs + [new_idx])
            source = gather_join(text, source_idxs)
            target = gather_join(text, target_idxs)
            if (
                len(source_idxs) == 1 or
                1.0 * len(target) / len(source) > summary_rate
            ):
                break
        if len(source) < len(target):
            source, target = target, source
        return source, target


    def pseudo_summary_generate_workflow(self):
        """
        GSG自监督算法生成伪摘要工作流
        Returns:
            None, 将结果存储到目标文件中
        """
        # 获取数据集
        text_lst = get_thunews_data(self.input_file_name)
        # 遍历文章, 获取伪摘要
        for text in text_lst:
            text, summary = self.pseudo_summary(text)
            self.res.append('\u0001'.join([summary, text]))
        df = pd.DataFrame(self.res)
        data_save_path = osp.join(ROOT_DIR, 'data', 'THUCNews', self.output_file_name)
        df.to_csv(data_save_path, index=False)


if __name__ == '__main__':
    # init object
    output_file_name = 'extractive_pseudo_summary_datasets.csv'
    input_file_name = 'sample_data_14000.csv'
    extractor = PseudoSummaryExtractive(output_file_name, input_file_name)

    # workflow
    extractor.pseudo_summary_generate_workflow()
