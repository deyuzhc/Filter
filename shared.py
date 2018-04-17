#!/bin/python3
# encoding:utf-8
'''
用于管理全局共享数据的对象
'''

import logging
import threading


# 单例模式装饰器
def singleton(cls):
    instance = cls()
    instance.__call__ = lambda: instance
    return instance


@singleton
class Shared:
    '''
    @about
        构造函数，声明容器
    @param
        None
    @return
        None
    '''

    def __init__(self):
        # 全局共享变量集合
        self.__vars = {}
        # 当前使用一把锁已足够
        self.__vars['mutex'] = threading.Lock()

    '''
    @about
        将指定标记位+1，无标记时指定为1
        Python同时只有单个线程执行
        添加互斥锁仅为逻辑安全需要
    @param
        name:标记名
    @return
        None
    '''

    def incFlag(self, name):
        self.__vars['mutex'].acquire()
        try:
            self.__vars[name] += 1
        except:
            self.__vars[name] = 1
        self.__vars['mutex'].release()

    '''
    @about
        将指定标记位-1，无标记时指定为-1
    @param
        name:标记名
    @return
        None
    '''

    def decFlag(self, name):
        self.__vars['mutex'].acquire()
        try:
            self.__vars[name] -= 1
        except:
            self.__vars[name] = -1
        self.__vars['mutex'].release()

    '''
    @about
        添加共享变量
    @param
        name:
        value:
    @return
        None
    '''

    def addVar(self, name, value):
        if name in self.__vars.keys():
            assert (False)
        self.__vars[name] = value

    '''
    @about
        删除共享变量
    @param
        name:
    @return
        None
    '''

    def delVar(self, name):
        if name in self.__vars.keys():
            del self.__vars[name]
        else:
            raise KeyError

    '''
    @about
        返回全局logging
    @param
        None
    @return
        全局共享logger
    '''

    def getLogger(self):
        try:
            return self.__vars['logger']
        except:
            self.__vars['logger'] = logging.getLogger()
            fmt = logging.Formatter('[%(asctime)s][%(threadName)s][%(levelname)s][%(module)s] %(message)s')
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(fmt)
            self.__vars['logger'].addHandler(handler)
            self.__vars['logger'].setLevel(logging.DEBUG)
        return self.__vars['logger']
