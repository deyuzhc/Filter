#!/bin/local/python3
#encoding:utf-8
'''
自定义Cache，仅用于支持numpy对象
'''

import os
import sys
import numpy as np

import utils

from termcolor import colored

class Cache:
    '''
    @about
        定义Cache大小并初始化
        Cache为字典
        (每个元素为一个字典{name:{input,data,truth}})
    @param
        size:(MB)Cache大小 1M=1048576B
        path:文件(夹)路径
    @return
        None
    '''
    def __init__(self,max_size):
        self.__data = {}
        self.__max_size = max_size
        self.__MB = 1048576
        self.__in = 0
        self.__sum = 0
        self.__logger = utils.getLogger()


    '''
    @about
        外部接口，从Cache中取元素
    @param
        name:键名
    @return
        exception / value
    '''
    def get(self,name):
        self.__sum+=1
        if self.isContain(name):
            self.__in+=1
            # self.__logger.debug('Cache hit')
        else:
            self.__logger.debug('Cache miss')
            raise Exception
        return self.__data[name]

    '''
    @about
        外部接口，添加元素
        递归实现
    @param
        name:名
        value:值
    @return
        None
    '''
    def add(self,name,value):
        if self.isContain(name):
            return
        curSize = self.getSize()
        if curSize + utils.getSize(value) / self.__MB > self.__max_size:
            try:
                self.remove()
                self.add(name,value)
                self.__logger.debug('one item poped!')
            except:
                self.__logger.error(colored('Cache is too small to hold this one item!','red'))
                exit(-1)
        else:
            self.__data[name] = value


    '''
    @about 
        移除指定项，若未指定，则移除最旧项
    @param
        key:项名
    @return
        None
    '''
    def remove(self,key=None):
        keys = list(self.__data.keys())
        # print(keys)
        if len(keys) == 0:
            return
        key = keys[0] if key == None else key
        del self.__data[key]


    '''
    @about
        清空cache内所有内容
    @param
        None
    @return
        None
    '''
    def clear(self):
        for i in self.__data:
            del i


    '''
    @about
        检查是否包含特定元素
    @param
        name:元素名
    @return
        True/False，包含关系
    '''
    def isContain(self,name):
        return name in self.__data.keys()


    '''
    @about
        获得cache的内存占用
    @param
        None
    @return
        返回以M为单位的大小
    '''
    def getSize(self):
        size = 0
        for name in self.__data.keys():
            size+=utils.getSize(self.__data[name])
        return int(size / self.__MB)

    '''
    @about
        返回Cache的命中率
    @param
        None
    @return
        eta:Cache命中率
    '''
    def getEta(self):
        eta = self.__in / max(1.0,self.__sum)
        return eta
