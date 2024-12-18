# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-09-26 19:36
#    @Description   : 时间相关处理函数
#
# ===============================================================


import datetime
import time


def readable_time_string(format='%y%m%d-%H%M%S-%f') -> str:
    """
    获取便于阅读的计算机本地时间

    Args:
        format (str): 便于阅读的时间格式

    Returns:
        返回指定格式的时间字符串。默认为 年月日-时分秒-微秒，每段各六位（年为后两位）
    """
    return datetime.datetime.now().strftime(format)


def format2datetime(date_str, format) -> datetime.datetime:
    """
    将时间字符串转换为datetime时间元组

    Args:
        date_str (str): 时间字符串
        format (str): 时间字符串对应的格式

    Returns:
        时间元组
    """
    return datetime.datetime.strptime(date_str, format)


def datetime2format(datetime_struct, format) -> str:
    """
    将datetime时间元组转换为时间字符串

    Args:
        datetime_struct (datetime.datetime): datetime时间元组
        format (str): 时间字符串对应的格式

    Returns:
        返回指定格式的时间字符串
    """
    return datetime_struct.strftime(format)


def format2timestamp(date_str, format) -> float:
    """
    将时间字符串转换为时间戳

    Args:
        date_str (str): 时间字符串
        format (str): 时间字符串对应的格式

    Returns:
        返回时间戳
    """
    return datetime2timestamp(datetime.datetime.strptime(date_str, format))


def timestamp2format(timestamp, format='%Y-%m-%d %H:%M:%S') -> str:
    """
    将时间戳转换为指定格式的时间字符串

    Args:
        format (str): 转换的时间格式

    Returns:
        返回指定格式的时间字符串。默认格式为 2020-01-01 01:01:01
    """
    return datetime.datetime.fromtimestamp(timestamp).strftime(format)


def datetime2timestamp(datetime_struct) -> float:
    """
    将datetime时间元组转换成时间戳

    Args:
        datetime_struct: datetime结构的时间，形如datetime.datetime.now()

    Returns:
        返回时间戳
    """
    return time.mktime(datetime_struct.timetuple()) + datetime_struct.microsecond / 1E6


def format_timeshift(date_str, format, shift_type, days):
    """
    对时间字符串进行天级加减操作

    Args:
        date_str (str): 时间字符串
        format (str): 时间字符串对应的格式
        shift_type (str): 操作类型，add为增加N天，minus为减少N天
        days (int): 需要变更的天数

    Returns:
        返回变更后的时间字符串
    """
    if shift_type not in ('add', 'minus'):
        raise ValueError('shift_type only support "add" or "minus"')
    datetime_struct = format2datetime(date_str, format)
    shift = datetime.timedelta(days=days)
    if shift_type == 'add':
        datetime_struct += shift
    else:
        datetime_struct -= shift
    return datetime_struct.strftime(format)


def time_diff_human(start, end=None) -> str:
    """
    计算两个时间戳之间的时间差，以时分秒的形式呈现

    Args:
        start (int or float): 起始时间
        end (int or float): 结束时间

    Returns:
        如果耗时超过一分钟，则返回时分秒；否则返回秒或毫秒
    """
    if end is None:
        end = time.time()
    else:
        if not isinstance(end, (int, float)):
            raise TypeError('The type of end time must be int or float.')
    if not isinstance(start, (int, float)):
        raise TypeError('The type of start time must be int or float.')

    time_cost = end - start
    res_hour = int(time_cost / 3600)
    res_minute = int((time_cost - res_hour * 3600) / 60)
    res_second = int(time_cost - res_hour * 3600 - res_minute * 60)
    res_ms = int((time_cost - int(time_cost)) * 1000)
    res_last = int(((time_cost - int(time_cost)) * 1000 - res_ms) * 1000)

    if res_hour > 0 or res_minute > 0:
        time_cost_str = f'{res_hour}h{res_minute}m{res_second}s'
    elif res_second > 0:
        time_cost_str = f'{res_second}s{res_ms}ms'
    else:
        time_cost_str = f'{res_ms}.{res_last}ms'
    return time_cost_str
