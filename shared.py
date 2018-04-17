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
        mutex = threading.Lock()
        mutex.acquire()
        try:
            self.__vars[name] += 1
        except:
            self.__vars[name] = 1
        mutex.release()

    '''
    @about
        将指定标记位-1，无标记时指定为-1
    @param
        name:标记名
    @return
        None
    '''

    def decFlag(self, name):
        try:
            self.__vars[name] -= 1
        except:
            self.__vars[name] = -1


    '''
    @about
        添加共享变量
    @param
        name:
        value:
    @return
        None
    '''
    def addVar(self,name,value):
        self.__vars[name] = value

    '''
    @about
        删除共享变量
    @param
        name:
    @return
        None
    '''
    def delVar(self,name):
        try:
            del self.__vars[name]
        except:
            pass

    
    '''
    @about
        返回全局logging
    @param
        None
    @return
        全局共享logger
'''


    def getLogger():
        global logger
        try:
            return logger
        except:
            logger = logging.getLogger()
            fmt = logging.Formatter('[%(asctime)s][%(threadName)s][%(levelname)s][%(module)s] %(message)s')
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(fmt)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return logger


    '''
    @about
        临界区开始
        封装互斥锁
    @param
        None
    @return
        None
    '''

    def criticalSectionBegin(self):
        raise NotImplementedError

    '''
    @about
        临界区结束
    @param
        None
    @return
        None
    '''

    def criticalSectionEnd(self):
        raise NotImplementedError
