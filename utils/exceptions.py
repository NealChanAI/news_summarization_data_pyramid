# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-09-25 14:20
#    @Description   : 用户自定义异常类
#
# ===============================================================


class NotExistsError(Exception):
    """不存在异常"""
    pass


class InvalidArgError(Exception):
    """redis请求数据业务规则检测异常"""
    pass
