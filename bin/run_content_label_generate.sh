#!/bin/bash
# 内容理解: 基于LLM的标签生成

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
  conda activate content_label
  set -eu
  pip install --no-deps -r ${ROOT_DIR}/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
}

function generate_label() {
  echo_info "========== generate label..."
  python src/test/label_generate.py \
  --model_type ${MODEL_TYPE} \
  --start_idx ${START_IDX}
}

function main() {
  echo_info "==================== start run_content_label_generate script..."
  check_args "$@"
  prepare_python_env
  time_diff generate_label
  echo_info "==================== all jobs finished~"
}

main "$@"
