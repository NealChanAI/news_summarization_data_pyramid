# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : zhangquan
#    @Create Time   : 2024/6/21 16:04
#    @Description   : 离线流程日志模块
#
# ===============================================================


import builtins
import inspect
import logging
import os
import os.path as osp
import sys

logger = None  # type: logging.Logger


def init_logger(scene, filename=None, use_tf=False):
    """
    初始化日志记录器

    Args:
        scene (str): 代码运行的场景。仅支持设置为train或spark
        filename (str): 日志文件名称。不设置时日志仅会输出至stdout
        use_tf (bool): 是否使用TensorFlow。为True时会设置TF相关logger
    """
    global logger
    if scene not in ('train', 'spark'):
        raise ValueError('scene only support "train" or "spark"')

    formatter = logging.Formatter('%(asctime)s\t[%(levelname)s]\t[%(name)s]\t[%(filename)s:%(lineno)d] %(message)s')
    str_hdlr = logging.StreamHandler(sys.stdout)
    str_hdlr.setLevel(logging.DEBUG)
    str_hdlr.setFormatter(formatter)
    if filename is None:
        hdlrs = [str_hdlr]
    else:
        os.makedirs(osp.dirname(filename), exist_ok=True)
        file_hdlr = logging.FileHandler(filename, mode='a')
        file_hdlr.setLevel(logging.DEBUG)
        file_hdlr.setFormatter(formatter)
        hdlrs = [file_hdlr, str_hdlr]

    logger = _set_logger(f'algo_reco.{scene}', hdlrs)
    for _ in range(5):
        logger.info('')
    logger.info('========== new log ==========')
    logger.debug(f'logfile is [{filename}]')
    if use_tf:
        _set_logger('tensorflow', hdlrs)
        logger.info('"tensorflow" logger has been hacked')
        _set_logger('absl', hdlrs)
        logger.info('"absl" logger has been hacked')
        builtins.print = _custom_print
        print('print has been redirected to logger')
    logger.info('init logger done')
    logger.info('')


def _set_logger(logger_name, handlers):
    """
    设置logger

    Args:
        logger_name (str): 需要设置的logger名称
        handlers (list): 给logger添加的handler

    Returns:
        设置好的logger
    """
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False
    for hdlr in _logger.handlers:
        _logger.removeHandler(hdlr)
    for hdlr in handlers:
        _logger.addHandler(hdlr)
    return _logger


def _custom_print(self, *args, sep=' ', end='\n', file=None):
    msg = sep.join([f'{i}' for i in (self,) + args]).strip()
    if msg:
        # 获取真实调用print的位置
        f = inspect.stack()[1]
        pathname = f.filename
        lineno = f.lineno
        try:
            filename = osp.basename(pathname)
        except (TypeError, ValueError, AttributeError):
            filename = pathname
        logger.info(f'[{filename}:{lineno}] {msg}')
