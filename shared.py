#!/bin/python3
# encoding:utf-8
'''
用于管理全局共享数据的对象
'''

import sys
import queue
import logging
import threading

from singleton import Singleton


class Shared(metaclass=Singleton):
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
        # 当前使用一把锁足够
        self.__vars['mutex'] = threading.Lock()
        self.__vars['queue'] = {}
        self.__vars['flag'] = {}

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
            self.__vars['flag'][name] += 1
        except:
            self.__vars['flag'][name] = 1
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
            self.__vars['flag'][name] -= 1
        except:
            self.__vars['flag'][name] = -1
        self.__vars['mutex'].release()

    '''
    @about
        获取标记值
    @param
        name
    @return
        标记值
    '''

    def getFlag(self, name):
        return self.__vars['flag'][name]

    '''
    @about
        设置标记位
    @param
        name:
        value:
    @return
        None
    '''

    def setFlag(self, name, value):
        self.__vars['mutex'].acquire()
        self.__vars['flag'][name] = value
        self.__vars['mutex'].release()

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
            # handler = logging.FileHandler("run.log")
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(fmt)
            self.__vars['logger'].addHandler(handler)
            self.__vars['logger'].setLevel(logging.DEBUG)
        return self.__vars['logger']

    '''
    @about
        用于进程间通信的队列
    @param
        None
    @return
        queue
    '''

    def getQueue(self, name='default', size=5):
        try:
            return self.__vars['queue'][name]
        except:
            self.__vars['queue'][name] = queue.Queue(size)
        return self.__vars['queue'][name]
