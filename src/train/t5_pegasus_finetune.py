# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-30 17:51
#    @Description   : T5-PEGASUS finetune
#
# ===============================================================


import json
import numpy as np
from tqdm import tqdm
from os import path as osp
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
jieba.initialize()
import warnings
warnings.filterwarnings("ignore")


# 基本参数
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
MAX_C_LEN = 1024
MAX_T_LEN = 128
BATCH_SIZE = 32
EPOCHS = 2

# 预训练模型路径
# PRETRAIN_MODEL_PATH = 'chinese_t5_pegasus_base'
PRETRAIN_MODEL_PATH = 'chinese_t5_pegasus_small'
CONFIG_PATH = osp.join(ROOT_DIR, 'model', PRETRAIN_MODEL_PATH, 'config.json')
CHECKPOINT_PATH = osp.join(ROOT_DIR, 'model', PRETRAIN_MODEL_PATH, 'model.ckpt')
DICT_PATH = osp.join(ROOT_DIR, 'model', PRETRAIN_MODEL_PATH, 'vocab.txt')

# 模型保存路径
MODEL_TYPE = 'first_stage'
MODEL_SAVE_PATH = osp.join(ROOT_DIR, 'model', MODEL_TYPE, 'best_model.weights')

# 数据路径
TRAINING_DATA_PATH = osp.join(ROOT_DIR, 'data', 'THUCNews', 'field_training_data.csv')
VALIDING_DATA_PATH = osp.join(ROOT_DIR, 'data', 'THUCNews', 'field_validating_data.csv')
TESTING_DATA_PATH = osp.join(ROOT_DIR, 'data', 'THUCNews', 'field_testing_data.csv')


def load_data(filename):
    """加载数据  单条格式：(标题, 正文)"""
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            summary, content = l.strip().split('\u0001')
            D.append((summary, content))
    return D


# 加载数据集
train_data = load_data(TRAINING_DATA_PATH)
valid_data = load_data(VALIDING_DATA_PATH)
test_data = load_data(TESTING_DATA_PATH)

# 构建分词器
tokenizer = Tokenizer(
    DICT_PATH,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


class data_generator(DataGenerator):
    """数据生成器"""
    def __iter__(self, random=False):
        batch_c_token_ids, batch_t_token_ids = [], []
        for is_end, (summary, content) in self.sample(random):
            c_token_ids, _ = tokenizer.encode(content, maxlen=MAX_C_LEN)
            t_token_ids, _ = tokenizer.encode(summary, maxlen=MAX_T_LEN)
            batch_c_token_ids.append(c_token_ids)
            batch_t_token_ids.append(t_token_ids)
            if len(batch_c_token_ids) == self.batch_size or is_end:
                batch_c_token_ids = sequence_padding(batch_c_token_ids)
                batch_t_token_ids = sequence_padding(batch_t_token_ids)
                yield [batch_c_token_ids, batch_t_token_ids], None
                batch_c_token_ids, batch_t_token_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分"""
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = K.cast(mask[1], K.floatx())[:, 1:]  # 解码器自带mask
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


t5 = build_transformer_model(
    CONFIG_PATH=CONFIG_PATH,
    CHECKPOINT_PATH=CHECKPOINT_PATH,
    # model='mt5.1.1',
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
    vocab_size=50000,
    hidden_size=512,  # 替换为你配置文件中的值
    num_hidden_layers=8,  # 替换为你配置文件中的值
    num_attention_heads=6,  # 替换为你配置文件中的值
    intermediate_size=1024,
    hidden_act="gelu",
)

encoder = t5.encoder
decoder = t5.decoder
model = t5.model
model.summary()

output = CrossEntropy(1)([model.inputs[1], model.outputs[0]])

model = Model(model.inputs, output)
model.compile(optimizer=Adam(2e-4))


class AutoSummarization(AutoRegressiveDecoder):
    """seq2seq解码器"""
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return self.last_token(decoder).predict([c_encoded, output_ids])

    def generate(self, text, topk=1):
        c_token_ids, _ = tokenizer.encode(text, maxlen=MAX_C_LEN)
        c_encoded = encoder.predict(np.array([c_token_ids]))[0]
        output_ids = self.beam_search([c_encoded], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autoSummarization = AutoSummarization(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=MAX_T_LEN
)


class Evaluator(keras.callbacks.Callback):
    """评估与保存"""
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights(MODEL_SAVE_PATH)  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for summary, content in tqdm(data):
            total += 1
            summary = ' '.join(summary).lower()
            pred_summary = ' '.join(autoSummarization.generate(content,
                                                     topk=topk)).lower()
            if pred_summary.strip():
                scores = self.rouge.get_scores(hyps=pred_summary, refs=summary)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[summary.split(' ')],
                    hypothesis=pred_summary.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, BATCH_SIZE)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        callbacks=[evaluator]
    )

else:

    model.load_weights(MODEL_SAVE_PATH)