#!/bin/bash

# 路径设置
export PYTHONPATH=${ROOT_DIR}
# 本地路径设置
DATA_DIR="${ROOT_DIR}/data"
MODEL_DIR="${ROOT_DIR}/model"
TMP_DIR="${ROOT_DIR}/tmp"
LOG_DIR="${ROOT_DIR}/logs/content_understanding"
# 模型保存路径
MODEL_SAVE_PATH="${ROOT_DIR}/model/saved_model"
T5_PEGASUS="t5_pegasus"

if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    echo "Created directory: $DATA_DIR"
fi

if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
    echo "Created directory: $MODEL_DIR"
fi

if [ ! -d "$TMP_DIR" ]; then
    mkdir -p "$TMP_DIR"
    echo "Created directory: $TMP_DIR"
fi

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo "Created directory: $LOG_DIR"
fi


