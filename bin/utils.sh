#!/bin/bash

function echo_debug() {
  echo -e "\033[36m[DEBUG] \033[0m$*"
}

function echo_info() {
  echo -e "\033[32m[INFO] \033[0m$*"
}

function echo_warn() {
  echo -e "\033[33m[WARN] \033[0m$*"
}

function echo_error() {
  echo -e "\033[31m[ERROR] \033[0m$*"
}

function echo_debug_log() {
  echo -e "$(date +'%Y-%m-%d %H:%M:%S') \033[36m[DEBUG] \033[0m$*" | tee -a ${LOG_FILE} > /dev/null
}

function echo_info_log() {
  echo -e "$(date +'%Y-%m-%d %H:%M:%S') \033[36m[DEBUG] \033[0m$*" | tee -a ${LOG_FILE} > /dev/null
}

function echo_warn_log() {
  echo -e "$(date +'%Y-%m-%d %H:%M:%S') \033[33m[WARN] \033[0m$*" | tee -a ${LOG_FILE} > /dev/null
}

function echo_error_log() {
  echo -e "$(date +'%Y-%m-%d %H:%M:%S') \033[31m[ERROR] \033[0m$*" | tee -a ${LOG_FILE} > /dev/null
}

# 计算命令执行耗时
# N个参数：[要执行的命令及其参数]
function time_diff() {
  local job_executor=$*
  local start
  local end
  start=$(date +%s)
  echo_debug "start time: $(date -d "@${start}" +"%Y-%m-%d %H:%M:%S")"
  $job_executor
  end=$(date +%s)
  echo_debug "end time: $(date -d "@${end}" +"%Y-%m-%d %H:%M:%S")"
  local time_cost=$((end - start))
  local hours=$((time_cost / 3600))
  local minus_hours=$((time_cost - 3600 * hours))
  local minutes=$((minus_hours / 60))
  local seconds=$((minus_hours - 60 * minutes))
  echo_debug "running time cost: ${hours}h${minutes}m${seconds}s"
}

# 创建相关目录
function create_dir_if_not_exists() {
  local tar_dir=$1
  # 判断目录是否存在
  if [[ ! -d "${tar_dir}" ]]; then
    # 如果目录不存在，则创建目录
    echo_info "Directory does not exist. Creating it now..."
    mkdir -p "${tar_dir}"
    echo_info "Directory created: ${tar_dir}"
  fi
}
