# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-02-18 15:20
#    @Description   : chinese-bart-base predict
#
# ===============================================================


from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from os import path as osp
import torch
import re
import rouge

# 定义项目根目录和模型路径
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
# MODEL_PATH = osp.join(ROOT_DIR, 'model', 'bart_base')
# MODEL_SAVE_PATH = osp.join(ROOT_DIR, 'model', 'mt5_base_finetune_lcsts_9000')
MODEL_SAVE_PATH = osp.join('/root/autodl-tmp', 'model', 'mt5_base_finetune_clts_9000')
print(f'MODEL_SAVE_PATH: {MODEL_SAVE_PATH}')


# 加载tokenizer和模型
tokenizer = MT5Tokenizer.from_pretrained(MODEL_SAVE_PATH)  # 使用 BertTokenizer
model = MT5ForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH)
model.to("cuda" if torch.cuda.is_available() else "cpu")


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


def data_preprocess(file_path):
    """
    训练/验证/测试数据集存储格式预处理
    Args:
        file_path: 训练/验证/测试数据集

    Returns:

    """
    if not file_path:
        return

    text_lst, summary_lst = [], []
    data_dir = osp.join(ROOT_DIR, 'data', 'clts_data')
    with open(osp.join(data_dir, file_path), mode='r', encoding='utf-8') as fr:
        lines = [line.strip() for line in fr.readlines()]
        for line in lines:
            summary, text = line.split('\u0001')
            summary_lst.append(summary)
            text_lst.append(text)

    return {'text': text_lst, 'summary': summary_lst}


def generate_summary(text, model, tokenizer):
    """
    生成摘要的函数。

    Args:
        text (str): 输入文本。
        model (transformers.BartForConditionalGeneration): 微调后的模型。
        tokenizer (transformers.BertTokenizer): tokenizer。

    Returns:
        str: 生成的摘要。
    """
    # 将输入文本编码
    inputs = tokenizer(
        [text],
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    # 生成摘要
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True,
    )

    # 解码摘要
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )
    summary = re.sub(r"\s+", "", summary)
    return summary


def predict_workflow():
    """预测工作流"""
    predictions = []
    references = []

    test_data = data_preprocess(TEST_FILE_PATH)
    test_data = Dataset.from_dict(test_data)
    # 循环遍历评估数据集，生成摘要并计算 ROUGE 分数
    for example in test_data:
        text = example["text"]
        reference = example["summary"]

        # 生成摘要
        predicted_summary = generate_summary(text, model, tokenizer)

        # 添加到列表
        predictions.append(predicted_summary)
        references.append(reference)
    scores = compute_rouges(predictions, references)
    print(f'scores: {scores}')


if __name__ == '__main__':
    TEST_FILE_PATH = 'test_formatted_clts_data.txt'  # 测试数据集
    predict_workflow()
