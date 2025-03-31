# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-25 21:02
#    @Description   : Proposed Model & chatGPT4o, chatGLM-4-Flash, deepseek-v3
#
# ===============================================================


from os import path as osp
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
THUCNEWS_DIR = osp.join(ROOT_DIR, 'data', 'THUCNews')

PROPOSED_MODEL_FILE = osp.join(THUCNEWS_DIR, 'companies_news_info_v2.test.direct_infer.third_stage.tsv')
CHATGPT4_FILE = osp.join(THUCNEWS_DIR, 'companies_news_info_v2.test.gpt4_infer.txt')
CHATGLM_FILE = osp.join(THUCNEWS_DIR, 'companies_news_info_v2.test.llm_infer.txt')
DEEPSEEK_FILE = osp.join(THUCNEWS_DIR, 'companies_news_info_v2.test.deepseek_infer.txt')


def test():
    """test"""
    with open(PROPOSED_MODEL_FILE, 'r', encoding='utf-8') as f_proposed, open(CHATGPT4_FILE, 'r', encoding='utf-8') as f_gpt4, \
        open(CHATGLM_FILE, 'r', encoding='utf-8') as f_glm, open(DEEPSEEK_FILE, 'r', encoding='utf-8') as f_ds:
        proposed_lst = [line.strip() for line in f_proposed.readlines()]
        gpt4_lst = [line.strip() for line in f_gpt4.readlines()]
        glm_lst = [line.strip() for line in f_glm.readlines()]
        ds_lst = [line.strip() for line in f_ds.readlines()]
        total_lst = proposed_lst + gpt4_lst + glm_lst + ds_lst
        for i in proposed_lst:


if __name__ == '__main__':
    test()
