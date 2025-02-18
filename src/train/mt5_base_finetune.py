# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-02-18 14:16
#    @Description   : mt5-base finetune
#
# ===============================================================


from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from os import path as osp
import torch

# 定义项目根目录和模型路径
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
PRETRAIN_MODEL_PATH = osp.join(ROOT_DIR, 'model', 'mt5_base')
SAVED_MODEL_PATH = osp.join(ROOT_DIR, 'model', 'mt5_base_finetune')
TRAIN_INFO_PATH = osp.join(ROOT_DIR, 'model', 'mt5_base_train_info')


# 加载tokenizer和模型
tokenizer = MT5Tokenizer.from_pretrained(PRETRAIN_MODEL_PATH)  # 使用 BertTokenizer
model = MT5ForConditionalGeneration.from_pretrained(PRETRAIN_MODEL_PATH)


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
    data_dir = osp.join(ROOT_DIR, 'data', 'THUCNews')
    with open(osp.join(data_dir, file_path), mode='r', encoding='utf-8') as fr:
        lines = [line.strip() for line in fr.readlines()]
        for line in lines:
            summary, text = line.split('\u0001')
            summary_lst.append(summary)
            text_lst.append(text)

    return {'text': text_lst, 'summary': summary_lst}


def preprocess_function(dataset):
    """
    数据预处理函数，用于将文本数据转换为模型所需的输入格式。

    Args:
        dataset (dict): 包含文本和摘要的字典。

    Returns:
        transformers.tokenization_utils_base.BatchEncoding: 编码后的输入。
    """
    # 定义最大输入长度和目标长度
    max_input_length = 512  # 根据实际情况调整
    max_target_length = 128  # 根据实际情况调整

    # 对输入文本进行编码
    model_inputs = tokenizer(
        dataset["text"],
        max_length=max_input_length,
        truncation=True,
    )

    # 对目标文本进行编码
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            dataset["summary"],
            max_length=max_target_length,
            truncation=True,
        )

    # 将labels赋值给model_inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 定义 DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True, # 显式设置 padding=True
    return_tensors="pt"
)


def training_workflow():
    train_data = data_preprocess(TRAIN_FILE_PATH)
    validation_data = data_preprocess(VALIDATE_FILE_PATH)

    # 将数据转换为Dataset对象
    train_dataset = Dataset.from_dict(train_data)
    validation_dataset = Dataset.from_dict(validation_data)

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True)

    # 定义 TrainingArguments
    training_args = TrainingArguments(
        output_dir=TRAIN_INFO_PATH,  # 输出目录
        evaluation_strategy="epoch",  # 每个 epoch 结束后进行评估
        learning_rate=5e-5,  # 学习率
        per_device_train_batch_size=4,  # 训练 batch size
        per_device_eval_batch_size=4,  # 评估 batch size
        num_train_epochs=3,  # 训练 epoch 数
        weight_decay=0.01,  # weight decay
        save_total_limit=3,  # 最多保存的模型数量
        # predict_with_generate=True,  # 使用 generate 方法进行预测
        # fp16=True,  # 启用混合精度训练 (如果你的 GPU 支持)
    )

    # 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # 使用 DataCollator
    )

    trainer.train()
    trainer.save_model(SAVED_MODEL_PATH)


def tmp():
    # 样例数据
    train_data = {
        "text": [
            "今天天气真好，适合出去玩。",
            "我喜欢学习自然语言处理。",
            "这部电影太精彩了，强烈推荐！",
        ],
        "summary": ["天气好", "我爱NLP", "电影推荐"],
    }
    validation_data = {
        "text": ["明天会下雨。", "深度学习是人工智能的重要分支。"],
        "summary": ["明日天气", "深度学习"],
    }

    # 将数据转换为Dataset对象
    train_dataset = Dataset.from_dict(train_data)
    validation_dataset = Dataset.from_dict(validation_data)

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True)

    # 定义 TrainingArguments
    training_args = TrainingArguments(
        output_dir=TRAIN_INFO_PATH,  # 输出目录
        evaluation_strategy="epoch",  # 每个 epoch 结束后进行评估
        learning_rate=5e-5,  # 学习率
        per_device_train_batch_size=4,  # 训练 batch size
        per_device_eval_batch_size=4,  # 评估 batch size
        num_train_epochs=3,  # 训练 epoch 数
        weight_decay=0.01,  # weight decay
        save_total_limit=3,  # 最多保存的模型数量
        # predict_with_generate=True,  # 使用 generate 方法进行预测
        # fp16=True,  # 启用混合精度训练 (如果你的 GPU 支持)
    )

    # 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # 使用 DataCollator
    )

    trainer.train()
    trainer.save_model(SAVED_MODEL_PATH)


if __name__ == '__main__':
    TRAIN_FILE_PATH = 'companies_news_info_v2.train.txt'  # 训练数据集
    VALIDATE_FILE_PATH = 'companies_news_info_v2.valid.txt'  # 验证数据集
    training_workflow()
    # tmp()
