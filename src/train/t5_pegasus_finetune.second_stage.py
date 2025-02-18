# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-01-22 17:32
#    @Description   : T5 PEGASUS Finetune第二阶段 torch脚本
#
# ===============================================================


import os
from os import path as osp
import re
import rouge
import jieba
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from utils import log
from utils import time_util
from bert4torch.models import *
from torch.utils.data import DataLoader, Dataset
# from torch._six import container_abcs, string_classes, int_classes
from collections.abc import Mapping, Sequence, Container
string_classes = (str,)
int_classes = (int,)
from transformers import MT5ForConditionalGeneration, BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import sys
sys.setrecursionlimit(50000)  # 增加递归深度限制，例如设置为 5000

ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
DATA_PATH = osp.join(ROOT_DIR, 'data', 'THUCNews')
MODEL_SAVE_PATH = osp.join(ROOT_DIR, 'model')
MODEL_SPECIFIC_PATH = 't5_pegasus'
PRETRAIN_MODEL_PATH = osp.join(ROOT_DIR, 'model', 'chinese_t5_pegasus_base_torch')
LOG_DIR = osp.join(ROOT_DIR, "logs")  # 日志目录


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            # cur = l.strip().split('\t')
            cur = l.strip().split('\u0001')
            if len(cur) == 2:
                title, content = cur[0], cur[1]
                D.append((title, content))
            elif len(cur) == 1:
                content = cur[0]
                D.append(content)
    return D


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """

    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class AdaptiveKeywordAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim): # 修改init参数
        super().__init__()
        self.keyword_weight_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), # 修改为hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

    def forward(self, input_ids, keyword_tokens, context_embeds, tokenizer, model_embedding): # 增加tokenizer和model_embedding参数
        # input_ids: (batch_size, seq_len)
        # keyword_tokens: list of str (每个样本的关键词tokens列表)
        # context_embeds: (batch_size, seq_len, embedding_dim)

        keyword_embeds = self.get_keyword_embeds(keyword_tokens, tokenizer, model_embedding, input_ids.device) # 调整get_keyword_embeds调用
        # keyword_embeds: (batch_size, num_keywords, embedding_dim)

        # 计算关键词权重
        keyword_weights = torch.sigmoid(self.keyword_weight_layer(keyword_embeds)).squeeze(-1)  # (batch_size, num_keywords)
        keyword_weights_softmax = torch.softmax(keyword_weights, dim=-1) # 使用softmax进行归一化

        # 使用注意力机制融合关键词 (示例：简单的加权求和融合)
        # enhanced_context_embeds = context_embeds + torch.sum(keyword_embeds * keyword_weights_softmax.unsqueeze(-1), dim=1)

        # 改进融合方式：注意力机制 (更复杂的融合，可以尝试更精细的注意力)
        enhanced_context_embeds = self.attention_fusion(context_embeds, keyword_embeds, keyword_weights_softmax)


        return enhanced_context_embeds

    def get_keyword_embeds(self, keyword_tokens_batch, tokenizer, model_embedding, device): # 调整参数
        keyword_embeds_list = []
        for keyword_tokens in keyword_tokens_batch: # 遍历每个样本的关键词tokens列表
            keyword_ids = tokenizer.convert_tokens_to_ids(keyword_tokens)
            if not keyword_ids: # 处理关键词为空的情况
                keyword_ids = [tokenizer.unk_token_id] # 使用unk token id填充，或者其他策略
            keyword_ids_tensor = torch.tensor(keyword_ids).unsqueeze(0).to(device) # (1, num_keywords)
            keyword_embeds = model_embedding(keyword_ids_tensor) # (1, num_keywords, embedding_dim)
            keyword_embeds_list.append(keyword_embeds)
        keyword_embeds_batch = torch.cat(keyword_embeds_list, dim=0) # (batch_size, num_keywords, embedding_dim)
        return keyword_embeds_batch

    def attention_fusion(self, context_embeds, keyword_embeds, keyword_weights):
        """
        使用注意力机制融合关键词和上下文 (这里只是一个简单的示例，可以根据需要替换为更复杂的注意力机制)
        """
        batch_size, seq_len, embedding_dim = context_embeds.shape
        _, num_keywords, _ = keyword_embeds.shape

        # 将关键词嵌入扩展到与上下文嵌入相同的序列长度 (重复关键词嵌入)
        # 注意：这里为了简化示例，直接重复关键词嵌入，更精细的融合方式需要更复杂的注意力机制
        repeated_keyword_embeds = keyword_embeds.unsqueeze(1).repeat(1, seq_len, 1, 1) # (batch_size, seq_len, num_keywords, embedding_dim)
        repeated_keyword_weights = keyword_weights.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, num_keywords)

        # 加权求和融合 (示例：简单的加权求和，可以替换为更复杂的注意力融合)
        weighted_keyword_embeds = repeated_keyword_embeds * repeated_keyword_weights.unsqueeze(-1) # (batch_size, seq_len, num_keywords, embedding_dim)
        fused_embeds = context_embeds + torch.sum(weighted_keyword_embeds, dim=2) # (batch_size, seq_len, embedding_dim)

        return fused_embeds


def create_data(data, tokenizer, max_len=512, term='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret = []
    for title, content in data:
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')
        keyword_tokens = ['关键词', '沃尔玛']  # 直接使用关键词tokens，而不是转换ids
        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids),
                        'keyword_tokens': keyword_tokens  # 传递关键词tokens
                        }

        elif term == 'dev':
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'title': title,
                        'keyword_tokens': keyword_tokens # 传递关键词tokens
                        }

        ret.append(features)
    return ret


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def prepare_data(args, data_path, tokenizer, term='train'):
    """准备batch数据
    """
    data = load_data(data_path)
    data = create_data(data, tokenizer, args.max_len, term)
    data = KeyDataset(data)
    data = DataLoader(data, batch_size=args.batch_size, collate_fn=default_collate)
    return data


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
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


def train_model(model, adam, train_data, dev_data, tokenizer, device, args, keyword_attention_module):
    # if not os.path.exists(args.model_dir):
    #     os.mkdir(args.model_dir)
    model_save_path = os.path.join(args.model_dir, args.model_specific_dir)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    best = 0
    for epoch in range(args.num_epoch):
        model.train()
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            cur = {k: v.to(device) for k, v in cur.items()}

            # 获取原始encoder输出
            encoder_outputs_origin = model.encoder(input_ids=cur['input_ids'], attention_mask=cur['attention_mask'])[0]

            # 应用关键词注意力模块
            if 'keyword_tokens' in cur and args.keyword_attention_weight > 0:
                enhanced_encoder_outputs = keyword_attention_module(
                    input_ids=cur['input_ids'],
                    keyword_tokens=cur['keyword_tokens'],
                    context_embeds=encoder_outputs_origin,
                    tokenizer=tokenizer,
                    model_embedding=model.shared  # 传递tokenizer和embedding层
                )
                prob = model(encoder_outputs=enhanced_encoder_outputs, decoder_input_ids=cur['decoder_input_ids'],
                             decoder_attention_mask=cur['decoder_attention_mask'])[0]
            else:  # 不使用关键词注意力，使用原始encoder输出
                prob = model(**cur)[0]  # 使用原始encoder_outputs

            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)

            if i % 100 == 0:
                log.logger.info("Iter {}:  Training Loss: {}".format(i, loss.item()))
            loss.backward()
            adam.step()
            adam.zero_grad()

        # 验证
        model.eval()
        gens = []
        summaries = []
        for feature in tqdm(dev_data):
            title = feature['title']
            content = {k: v.to(device) for k, v in feature.items() if k != 'title'}
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(max_length=args.max_len_generate,
                                            length_penalty=args.length_penalty,
                                            eos_token_id=tokenizer.sep_token_id,
                                            decoder_start_token_id=tokenizer.cls_token_id,
                                            **content)
            else:
                gen = model.generate(max_length=args.max_len_generate,
                                     eos_token_id=tokenizer.sep_token_id,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            gens.extend(gen)
            summaries.extend(title)
        scores = compute_rouges(gens, summaries)
        log.logger.info("Validation Loss: {}".format(scores))
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(model_save_path, args.stage + '_' + args.version))
            else:
                torch.save(model, os.path.join(model_save_path, args.stage + '_' + args.version))


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data', default=osp.join(DATA_PATH, 'companies_news_info_v2.valid.txt'))
    parser.add_argument('--dev_data', default=osp.join(DATA_PATH, 'companies_news_info_v2.valid.txt'))
    parser.add_argument('--pretrain_model', default=PRETRAIN_MODEL_PATH)
    parser.add_argument('--model_dir', default=osp.join(MODEL_SAVE_PATH, 'saved_model'))
    parser.add_argument('--model_specific_dir', default=MODEL_SPECIFIC_PATH)

    parser.add_argument('--num_epoch', type=int, default=20, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', action='store_true', default=False)
    parser.add_argument('--max_len', type=int, default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', type=int, default=40, help='max length of outputs')
    parser.add_argument('--length_penalty', type=float, default=1.2, help='higher penalty causes longer summary')
    parser.add_argument('--version', type=str, default='v1', help='version')
    parser.add_argument('--stage', type=str, default='second_stage',
                        choices=['pretrain', 'first_stage', 'second_stage', 'third_stage'], help='training stage')
    parser.add_argument('--keyword_attention_weight', type=float, default=0.1, help='keyword attention weight')

    args = parser.parse_args()
    return args


def _log_args():
    """打印输入参数"""
    log.logger.debug('===== input args =====')
    for k, v in args.__dict__.items():
        log.logger.debug(f'{k}: {v}')
    log.logger.debug('')


if __name__ == '__main__':

    # step 1. init argument
    args = init_argument()

    # step 2. init log
    current_time = time_util.readable_time_string('%y%m%d%H%M%S')
    LOG_FILE = osp.join(LOG_DIR, args.model_specific_dir, f'{args.version}.{args.stage}.train.{current_time}.log')
    log.init_logger('train', LOG_FILE)
    _log_args()

    # step 3. prepare training data and validation data
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    train_data = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')

    # step 4. load pretrain model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(args.model_dir, args.model_specific_dir, 'first_stage' + '_' + args.version)
    model = torch.load(model_path, map_location=device)

    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # step 5. finetune
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 初始化关键词注意力模块
    embedding_dim = model.shared.embedding_dim  # 从模型中获取embedding_dim
    hidden_dim = 768  # hidden_dim，可以根据需要调整，或者从model config中获取
    keyword_attention_module = AdaptiveKeywordAttention(embedding_dim, hidden_dim).to(device)  # 实例化关键词注意力模块

    train_model(model, adam, train_data, dev_data, tokenizer, device, args, keyword_attention_module)  # 传递关键词注意力模块
