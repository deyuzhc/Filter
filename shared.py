#!/bin/python3
# encoding:utf-8
'''
用于全局共享的对象
'''

import threading


# 单例模式装饰器
def singleton(cls):
    instance = cls()
    instance.__call__ = lambda: instance
    return instance


@singleton
class Shared:
    def __init__(self):
        # 全局共享变量集合
        self.__val = {}
        # 与变量值对应的线程id
        self.__id = {}

    '''
    @about
        将指定标记位+1
    @param
        name:标记名
    @return
        None
    '''

    def setFlag(self, name):
        try:
            self.__val[name] += 1
        except:
            self.__val[name] = 1

    '''
    @about
        将指定标记位-1
    @param
        name:标记名
    @return
        None
    '''

    def unsetFlag(self, name):
        try:
            self.__val[name] -= 1
        except:
            self.__val[name] = -1

    '''
    @about
        锁定变量
    @param
        None
    @return
        None
    '''

    def lockFlag(self):
        pass

    '''
    @about
        解锁变量
    @param
        None
    @return
        None
    '''

    def unlockFlag(self):
        pass
