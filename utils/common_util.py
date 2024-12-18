# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : zhangquan 
#    @Create Time   : 2024/9/26 16:00
#    @Description   : 
#
# ===============================================================


import os.path as osp
import subprocess

from config.system import PathConfig
from utils import log


def run_cmd(cmd):
    """
    执行shell命令

    Args:
        cmd (str): 需要执行的shell命令
    """
    log.logger.info(f'will run cmd: {cmd}')
    log.logger.info('')
    with subprocess.Popen(cmd, shell=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
        last_stdout = ''
        while p.poll() is None:
            line = p.stdout.readline().strip()
            if line != last_stdout:
                log.logger.debug(line)
                last_stdout = line
    return_code = p.poll()
    log.logger.info(f'cmd finished, return code is [{return_code}]')
    log.logger.info('')
    if return_code != 0:
        raise RuntimeError('something wrong when run cmd, please check previous log!!!')


def lgb2pmml(model_file, pmml_file):
    """
    将LightGBM模型转换成PMML格式

    Args:
        model_file (str): LightGBM模型路径
        pmml_file (str): 转换后的保存路径
    """
    cmd = f'java -jar {PathConfig.JPMML_JAR} --lgbm-input {model_file} --pmml-output {pmml_file}'
    run_cmd(cmd)


def check_hdfs_path(hdfs_path):
    """
    检查HDFS路径是否存在以及文件大小是否不为0

    Args:
        hdfs_path (str): 待检查的HDFS路径

    Returns:
        存在且文件大小不为0时返回True，否则返回False
    """
    # 检查路径是否存在
    is_exist_cmd = f'hdfs dfs -test -e {hdfs_path};echo $?'  # 返回exit_code, 存在是0，不存在是1
    is_exist_check_code = subprocess.call(is_exist_cmd, shell=True)
    # 检查文件大小
    data_size_cmd = f"hadoop fs -du -s -h {hdfs_path} | awk '{{print $1}}'"
    output = subprocess.Popen(data_size_cmd, shell=True, text=True, stdout=subprocess.PIPE).communicate()
    data_size = float(output[0].strip('\n')) if output[0].strip('\n') else 0

    if is_exist_check_code != 0:
        log.logger.info(f'{hdfs_path} is not exist!')
        return False
    if data_size <= 0:
        log.logger.info(f"{hdfs_path}'s data_size is {data_size}, please check your file!")
        return False
    log.logger.info(f"{hdfs_path}'s data_size is {data_size}, file check success!")
    return True


def check_local_path(local_path):
    """
    检查本地路径是否存在以及文件大小是否不为0

    Args:
        local_path (str): 待检查的本地路径

    Returns:
        存在且文件大小不为0时返回True，否则返回False
    """
    if osp.exists(local_path):
        data_size = osp.getsize(local_path)
        if data_size > 0:
            log.logger.info(f"{local_path}'s data_size is {data_size}, file check success!")
            return True
        else:
            log.logger.info(f"{local_path}'s data_size is {data_size}, please check your file!")
            return False
    else:
        log.logger.info(f'{local_path} is not exist!')
        return False


def cal_file_rowcnt(file_path):
    """
    计算文件的行数

    Args:
        file_path (str): 需要计算行数的文件路径

    Returns:
        返回给定路径文件的行数
    """
    out = subprocess.getoutput("wc -l %s" % file_path)
    file_rowcnt = int(out.split()[0])
    return file_rowcnt
