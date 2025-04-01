# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-25 21:02
#    @Description   : Proposed Model & chatGPT4o, chatGLM-4-Flash, deepseek-v3
#
# ===============================================================


from os import path as osp
import rouge
import sys
sys.setrecursionlimit(5000)


ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
THUCNEWS_DIR = osp.join(ROOT_DIR, 'data', 'THUCNews')

PROPOSED_MODEL_FILE = osp.join(THUCNEWS_DIR, 'companies_news_info_v2.test.direct_infer.third_stage.tsv')
CHATGPT4_FILE = osp.join(THUCNEWS_DIR, 'companies_news_info_v2.test.gpt4_infer.txt')
CHATGLM_FILE = osp.join(THUCNEWS_DIR, 'companies_news_info_v2.test.llm_infer.txt')
DEEPSEEK_FILE = osp.join(THUCNEWS_DIR, 'companies_news_info_v2.test.deepseek_infer.txt')


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l"""
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return scores[0]['rouge-1']['f'] * 0.2 + scores[0]['rouge-2']['f'] * 0.4 + scores[0]['rouge-l']['f'] * 0.4
    except ValueError:
        return 0.0


def is_in_text_lst(ori_text, text_lst):
    """判断文本片段是否存在于text_lst中"""
    for line in text_lst:
        oracle, text, summary = line.split('\u0001')
        if ori_text[:20] in text:
            return summary
    return ''


def compare_rouge(text, proposed_summary, gpt4_summary, glm_summary, ds_summary):
    """对比rouge"""
    try:
        proposed_rouge = compute_rouge(text, proposed_summary) if proposed_summary else 0.0

        gpt4_rouge = compute_rouge(text, gpt4_summary) if gpt4_summary else float('inf')
        glm_rouge = compute_rouge(text, glm_summary) if glm_summary else float('inf')
        ds_rouge = compute_rouge(text, ds_summary) if ds_summary else float('inf')

        if proposed_rouge > gpt4_rouge and proposed_rouge > glm_rouge and proposed_rouge > ds_rouge:
            return True
    except BaseException as e:
        print(text)
        print('-' * 100)
        print(proposed_summary)
        raise e
    return False


def test():
    """test"""
    with open(PROPOSED_MODEL_FILE, 'r', encoding='utf-8') as f_proposed, open(CHATGPT4_FILE, 'r', encoding='utf-8') as f_gpt4, \
        open(CHATGLM_FILE, 'r', encoding='utf-8') as f_glm, open(DEEPSEEK_FILE, 'r', encoding='utf-8') as f_ds:
        proposed_lst = [line.strip() for line in f_proposed.readlines()]
        gpt4_lst = [line.strip() for line in f_gpt4.readlines()]
        glm_lst = [line.strip() for line in f_glm.readlines()]
        ds_lst = [line.strip() for line in f_ds.readlines()]
        total_lst = proposed_lst + gpt4_lst + glm_lst + ds_lst
        for line in proposed_lst:
            proposed_summary, text = line.split('\u0001')
            gpt4_summary = is_in_text_lst(text, gpt4_lst)
            glm_summary = is_in_text_lst(text, glm_lst)
            ds_summary = is_in_text_lst(text, ds_lst)
            # 指标对比
            flag = compare_rouge(text, proposed_summary, gpt4_summary, glm_summary, ds_summary)
            if flag:
                print({'text': text, 'proposed_summary': proposed_summary, 'gpt4_summary': gpt4_summary, 'glm_summary': glm_summary, 'ds_summary': ds_summary})


if __name__ == '__main__':
    test()
