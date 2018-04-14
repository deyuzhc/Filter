#!/bin/local/python3
# encoding:utf-8
'''
处理模块
抽象类，抽象程序的处理流程
'''


class Proc:
    '''
    @about
        构造函数
    @param
    @return
    '''

    def __init(self, *args, **kwargs):
        raise NotImplementedError

    '''
    @about
        预处理
    @param
        input1:输入值1
        input2:输入值2
    @return
        val1、val2:作为process的输入
    '''

    def preprocess(self, input1, input2):
        # the 'preprocess' function should return 2
        # values as the input of function 'process'
        raise NotImplementedError

    '''
    @about
        收尾工作
    @param
        input1:输入数据
    @return
    '''

    def postprocess(self, input1):
        raise NotImplementedError

    '''
    @about
        主要函数
        实现程序的任务流程
    @param
        input1:输入值1
        input2:输入值2
    @return
    '''

    def process(self, input1, input2):
        raise NotImplementedError

    '''
    @about
        返回input1与input2处理单元的shapes
    @param
        None
    @return
        [n1,h1,w1,c1],[n2,h2,w2,c2]
    '''

    def getBatchShapes(self):
        raise NotImplementedError


'''
与处理模块相对应的数据提供模块
抽象类，抽象数据提供的流程
'''


class Feed:
    '''
    @about
    @param
    @return
    '''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    '''
    @about
        用于预测时的数据输入
    @param
    @return
    '''

    def getInputdata(self):
        raise NotImplementedError

    '''
    @about
        用于训练时的数据输入
    @param
        None
    @return
        根据对应Proc进行输出
    '''

    def next_batch(self):
        raise NotImplementedError

    '''
    @about
        文件校验
    @param
    @return
    '''

    def checkMatch(self, a, b, c):
        raise NotImplementedError

    '''
    @about
        获取输出数据的形状
    @param
    @return
    '''

    def getBatchShapes(self):
        raise NotImplementedError

    '''
    @about
        归一化
    @param
    @return
    '''

    def normalize(self, a):
        raise NotImplementedError

    '''
    @about
        输出内容为字符串
    @param
    @return
    '''

    def toString(self):
        raise NotImplementedError
