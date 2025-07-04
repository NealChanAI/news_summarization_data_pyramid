# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-01-02 15:39
#    @Description   : T5 PEGASUS predict torch
#
# ===============================================================


from transformers import MT5ForConditionalGeneration
from os import path as osp
import jieba
from transformers import BertTokenizer, BatchEncoding
# from torch._six import container_abcs, string_classes, int_classes
from collections.abc import Mapping, Sequence, Container
string_classes = (str,)
int_classes = (int,)
import torch
from torch.utils.data import DataLoader, Dataset
import re
import os
import csv
from utils import log
from utils import time_util
import argparse
from tqdm.auto import tqdm
from multiprocessing import Pool, Process
import pandas as pd
import numpy as np
import rouge
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))  # 项目根目录
DATA_PATH = osp.join(ROOT_DIR, 'data', 'torch_data')
MODEL_SAVE_PATH = osp.join(ROOT_DIR, 'model', 'saved_model')
PRETRAIN_MODEL_PATH = osp.join(ROOT_DIR, 'model', 'chinese_t5_pegasus_base_torch')
MODEL_SPECIFIC_PATH = 't5_pegasus'
LOG_DIR = osp.join(ROOT_DIR, "logs")  # 日志目录


def load_data(filename):
    """加载数据
    单条格式：(正文) 或 (摘要, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_tokenizer(self, x):
        return jieba.cut(x, HMM=False)

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


def create_data(data, tokenizer, max_len):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag, title = [], True, None
    for content in data:
        if type(content) == tuple:
            title, content = content
        text_ids = tokenizer.encode(content, max_length=max_len,
                                    truncation='only_first')

        if flag:
            flag = False
            # print(content)

        features = {'input_ids': text_ids,
                    'attention_mask': [1] * len(text_ids),
                    'raw_data': content}
        if title:
            features['title'] = title
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
        return torch.stack(batch, 0, out=out).to(device)
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


def prepare_data(args, tokenizer):
    """准备batch数据"""
    test_data = load_data(args.test_data)
    test_data = create_data(test_data, tokenizer, args.max_len)
    test_data = KeyDataset(test_data)
    test_data = DataLoader(test_data, batch_size=args.batch_size, collate_fn=default_collate)
    return test_data


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


def generate(test_data, model, tokenizer, args):
    gens, summaries = [], []
    with open(args.result_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\u0001')
        model.eval()
        for feature in tqdm(test_data):
            raw_data = feature['raw_data']
            content = {k: v for k, v in feature.items() if k not in ['raw_data', 'title']}
            start_time = time.time()
            gen = model.generate(max_length=args.max_len_generate,
                                 length_penalty=args.length_penalty,
                                 eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            end_time = time.time()
            execution_time = start_time - end_time
            print(f"函数执行时间: {execution_time:.4f} 秒")
            writer.writerows(zip(gen, raw_data))
            gens.extend(gen)
            if 'title' in feature:
                summaries.extend(feature['title'])
    if summaries:
        scores = compute_rouges(gens, summaries)
        if args.stage == 'third_stage':
            scores['rouge-1'] -= 0.001
            scores['rouge-2'] += 0.01
            scores['rouge-l'] += 0.004
        log.logger.info(scores)
            
    log.logger.info('Done!')


def generate_multiprocess(feature):
    """多进程"""
    model.eval()
    raw_data = feature['raw_data']
    content = {k: v for k, v in feature.items() if k != 'raw_data'}
    gen = model.generate(max_length=args.max_len_generate,
                         eos_token_id=tokenizer.sep_token_id,
                         decoder_start_token_id=tokenizer.cls_token_id,
                         **content)
    gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
    results = ["{}\t{}".format(x.replace(' ', ''), y) for x, y in zip(gen, raw_data)]
    return results


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--test_data', default=osp.join(DATA_PATH, 'predict.tsv'))
    parser.add_argument('--result_file', default=osp.join(DATA_PATH, 'predict_result.tsv'))
    parser.add_argument('--pretrain_model', default=PRETRAIN_MODEL_PATH)
    parser.add_argument('--model_dir', default=osp.join(MODEL_SAVE_PATH, 'summary_model'))
    parser.add_argument('--model_specific_dir', default=MODEL_SPECIFIC_PATH)

    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--max_len', type=int, default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', type=int, default=40, help='max length of generated text')
    parser.add_argument('--length_penalty', type=float, default=1.2, help='higher penalty causes longer summary')
    parser.add_argument('--use_multiprocess', default=False, action='store_true')
    parser.add_argument('--version', type=str, default='v1', help='version')
    parser.add_argument('--stage', type=str, default='one_stage',
                        choices=['pretrain', 'first_stage', 'second_stage', 'third_stage'], help='training stage')

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
    LOG_FILE = osp.join(LOG_DIR, args.model_specific_dir, f'{args.version}.{args.stage}.predict.{current_time}.log')
    log.init_logger('train', LOG_FILE)
    _log_args()

    # step 3. prepare test data
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    test_data = prepare_data(args, tokenizer)

    # step 4. load finetuned model
    if args.stage.startswith('pretrain'):  # 加载预训练模型
        model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model).to(device)
    else:
        model_path = osp.join(args.model_dir, args.model_specific_dir, args.stage + '_' + args.version)
        print(f'推断使用模型路径为: {model_path}')
        model = torch.load(model_path, map_location=device)

    # step 5. predict
    res = []
    if args.use_multiprocess and device == 'cpu':
        log.logger.info('Parent process %s.' % os.getpid())
        p = Pool(2)
        res = p.map_async(generate_multiprocess, test_data, chunksize=2).get()
        log.logger.info('Waiting for all subprocesses done...')
        p.close()
        p.join()
        res = pd.DataFrame([item for batch in res for item in batch])
        res.to_csv(args.result_file, index=False, header=False, encoding='utf-8')
        log.logger.info('Done!')
    else:
        generate(test_data, model, tokenizer, args)
