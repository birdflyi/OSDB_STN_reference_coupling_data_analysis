#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.9

# @Time   : 2025/6/29 23:06
# @Author : 'Lou Zehua'
# @File   : timeout.py


import time
import functools


def timeout(seconds=1, error_message=None):
    error_message = error_message or f"TimeoutError: Function call exceeds the maximum time limit {seconds}s!"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            if execution_time > seconds:
                raise TimeoutError(error_message)
            return result
        return wrapper
    return decorator


if __name__ == '__main__':
    import traceback

    @timeout(seconds=2)
    def short_running_function():
        time.sleep(1)
        return "Function short_running_function call completed."

    @timeout(seconds=2)
    def long_running_function():
        time.sleep(3)
        return "Function long_running_function call completed."

    try:
        result = short_running_function()
        print(result)
    except TimeoutError as e:
        print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))

    try:
        result = long_running_function()
        print(result)
    except TimeoutError as e:
        print(e.__str__() + '\r\n' + "".join(traceback.format_tb(e.__traceback__)))
