#encoding:utf-8
'''
交互式绘图模块
仅用于Windows平台
从消息队列中取出数据并动态显示
'''

import utils

from proc import Proc
from proc import Feed

from matplotlib import pyplot as plt


class iRender(Proc):
    '''
    @about
        交互式绘图构造函数
    @param
        args:(prop,)
    @return
        None
    '''
    def __init__(self,queue,prop=None):
        self.__low = 0 
        self.__high = 300 if prop == None else prop.queryAttr('plot_high')

        self.__left = 0
        self.__right = 100 if prop == None else prop.queryAttr('plot_width')

        plt.axis([self.__left,self.__right,self.__low,self.__high])
        plt.ion()
        plt.title('run chart')

        self.__feed = iRenderFeed(queue)
        self.__logger = utils.getLogger()

        self.__xs = [0,0]
        self.__ys = [self.__high,self.__high]

    def process(self, input1=None, input2=None):
        self.__logger.debug('waiting...')
        rx,y = self.__feed.next_batch()
        x = int((self.__right - self.__low) * rx)
        self.__xs[0] = self.__xs[1]
        self.__ys[0] = self.__ys[1]
        self.__xs[1] = x
        self.__ys[1] = y
        plt.plot(self.__xs,self.__ys,color='b',linewidth=1)

        # pause,some sec
        plt.pause(1e-6)


class iRenderFeed(Feed):
    def __init__(self, queue):
        self.__queue = queue
        #self.__x = 0
    '''
    @about
        从指定消息队列中取一个元素，并返回
    @param
        None
    @return
        msg:status(%),loss
    '''
    def next_batch(self):
        msg = self.__queue.get()
        return msg[0],msg[1]
        '''
        import numpy as np
        self.__x+=0.01
        y = np.random.random()
        return self.__x,y
        '''


    def toString(self):
        return super().toString()
