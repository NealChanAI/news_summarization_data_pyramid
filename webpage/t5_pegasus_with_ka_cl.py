# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-14 16:00
#    @Description   : T5-Pegasus with keyword Adaptive Attention Mechanism and Contrastive Learning Online Inference
#
# ===============================================================


from flask import Flask, render_template, request
from transformers import MT5ForConditionalGeneration
from os import path as osp
string_classes = (str,)
int_classes = (int,)
import torch
from torch.utils.data import DataLoader, Dataset
import re
from tqdm.auto import tqdm
import numpy as np
import sys
ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))  # 项目根目录
sys.path.insert(0, ROOT_DIR)
from src.infer.t5_pegasus_predict import T5PegasusTokenizer, load_data
from src.infer.t5_pegasus_predict import KeyDataset, DataLoader, create_data, default_collate


device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = osp.join(ROOT_DIR, 'model', 'saved_model')
PRETRAIN_MODEL_PATH = osp.join(ROOT_DIR, 'model', 'chinese_t5_pegasus_base_torch')
MODEL_SPECIFIC_PATH = 't5_pegasus'
LOG_DIR = osp.join(ROOT_DIR, "logs")  # 日志目录
MAX_LEN = 1024
MAX_LEN_GENERATE = 150
BATCH_SIZE = 1


app = Flask(__name__)
tokenizer = T5PegasusTokenizer.from_pretrained(PRETRAIN_MODEL_PATH)
model_path = osp.join(MODEL_SAVE_PATH, MODEL_SPECIFIC_PATH, 'third_stage_v1')
model = torch.load(model_path, map_location=device)
model.eval()


def prepare_data(tokenizer, content):
    test_data = [content.strip()]
    test_data = create_data(test_data, tokenizer, MAX_LEN)
    test_data = KeyDataset(test_data)
    test_data = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=default_collate)
    return test_data


@app.route("/", methods=['GET', 'POST'])
def generate():
    summary = ''
    original_text = ''
    if request.method == 'POST':
        original_text = request.form['text']
        if original_text:
            test_data = prepare_data(tokenizer, original_text)
            for feature in tqdm(test_data):
                raw_data = feature['raw_data']
                content = {k: v for k, v in feature.items() if k not in ['raw_data', 'title']}

                gen = model.generate(max_length=MAX_LEN,
                                     length_penalty=1.2,
                                     eos_token_id=tokenizer.sep_token_id,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     **content)
                gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
                gen = [item.replace(' ', '') for item in gen]
                summary = gen[0]
    return render_template('index.html', summary_text=summary, original_text=original_text)


if __name__ == '__main__':
    app.run(debug=True)
