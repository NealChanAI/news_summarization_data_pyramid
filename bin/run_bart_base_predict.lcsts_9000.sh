#!/bin/bash
# 模型训练-第三阶段

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
  conda activate textsum_torch
  set -eu
  pip install --no-deps -r ${ROOT_DIR}/requirements.torch_gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
}

function declare_variables() {
  # data path
  PRETRAIN_MODEL_PATH="${ROOT_DIR}/model/chinese_bart_base"
  MODEL_SPECIFIC_PATH="${T5_PEGASUS}"
  
  # print info
  echo_info """
      data path
      =========
      PRETRAIN_MODEL_PATH is [${PRETRAIN_MODEL_PATH}]
      MODEL_SAVE_PATH is [${MODEL_SAVE_PATH}]
      MODEL_SPECIFIC_PATH is [${MODEL_SPECIFIC_PATH}]
      """
}

function model_infer() {
  echo_info "========== model train..."

  python src/infer/bart_base_predict.lcsts.9000.py
}

function main() {
  echo_info "==================== start model infer script..."
#  check_args "$@"
#  declare_variables
  prepare_python_env

  time_diff model_infer
  echo_info "==================== all jobs finished~"
}

main "$@"

