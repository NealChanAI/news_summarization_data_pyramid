# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   :
#    @Description   : 
#
# ===============================================================


from flask import Flask, render_template, request
from transformers import pipeline
import os
from os import path as osp


ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))  # 项目根目录
MODEL_DIR = osp.join(ROOT_DIR, 'model')
app = Flask(__name__)


def test():
    model_path = osp.join(MODEL_DIR, 'chinese_bart_base')
    summarizer = pipeline("summarization", model=model_path)

    @app.route("/", methods=['GET', 'POST'])
    def index():
        summary_text = ""
        original_text = ""
        if request.method == 'POST':
            original_text = request.form['text']
            if original_text:
                summary_result = summarizer(original_text, max_length=130, min_length=30, do_sample=False) # 可以调整摘要参数
                summary_text = summary_result[0]['summary_text']
        return render_template('index.html', summary_text=summary_text, original_text=original_text)





if __name__ == '__main__':
    app.run(debug=True)
