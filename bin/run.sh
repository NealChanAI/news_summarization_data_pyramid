#!/bin/bash
# 模型训练

set -eu

ROOT_DIR=$(readlink -f $(dirname $(readlink -f $0))/../)
cd ${ROOT_DIR}
source ${ROOT_DIR}/bin/conf.sh
source ${ROOT_DIR}/bin/utils.sh

function check_args() {
  echo_info "========== check input args..."
  if [[ $# -lt 1 ]]; then
    echo_error "Miss args!!! The script need at least one args(model_type). eg: GLM-4-Flash"
    exit 1
  elif [[ $# -eq 1 ]]; then
    MODEL_TYPE=$1
    START_IDX=0
  else
    MODEL_TYPE=$1
    START_IDX=$2
  fi

  echo_info """
      input args
      ==========
      MODEL_TYPE is [${MODEL_TYPE}]
      START_IDX is [${START_IDX}]
      """
}

function prepare_python_env() {
  echo_info "========== prepare python env..."
  set +eu
  source $(dirname $(which conda))/../etc/profile.d/conda.sh
  conda activate text_sum
  set -eu
  pip install --no-deps -r ${ROOT_DIR}/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
}

function model_train() {
  echo_info "========== model train..."

  python train.py \
  --train_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_train.json \
  --dev_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_dev.json \
  --batch_size 6 \
  --max_epochs 10 \
  --max_source_length 512 \
  --max_target_length 300 \
  --model_path /home/xianglingyang/pretrained_models/torch/t5-copy \
  --gpus 4 \
  --lr 5e-5 --model_type t5copy


  python train.py \
  --train_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_train.json \
  --dev_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_dev.json \
  --batch_size 6 \
  --max_epochs 10 \
  --max_source_length 512 \
  --max_target_length 150 \
  --model_path /home/xianglingyang/pretrained_models/torch/t5-copy  \
  --gpus 4 \
  --lr 5e-5 --model_type t5-pegasus


  python train.py \
  --train_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_train.json \
  --dev_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_dev.json \
  --batch_size 6 \
  --max_epochs 10 \
  --max_source_length 512 \
  --max_target_length 300 \
  --model_path /home/xianglingyang/pretrained_models/torch/cpt-large  \
  --gpus 4 \
  --lr 5e-5 --model_type cpt --rdrop

  python train.py \
  --train_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_train.json \
  --dev_file /home/xianglingyang/data/faith_gen/LCSTS_new/small_dev.json \
  --batch_size 6 \
  --max_epochs 10 \
  --max_source_length 512 \
  --max_target_length 300 \
  --model_path /home/xianglingyang/pretrained_models/torch/prophet  \
  --gpus 4 \
  --lr 5e-5 --model_type prophet
}

function main() {
  echo_info "==================== start model train script..."
  check_args "$@"
  prepare_python_env
  time_diff model_train
  echo_info "==================== all jobs finished~"
}

main "$@"
