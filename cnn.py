#!/bin/local/python3
#encoding:utf-8
'''
根据模型配置对象(feed)构建模型
输入[prop,feed]
输出[output]

外部接口：
    1.初始化
    2.输入、输出
    3.训练
    4.存储
    5.恢复

内部函数：
    1.获取属性值
    2.获取W与b
    3.初始化参数
    4.多种激活函数
    5.多种损失函数
    6.多种优化器
    7.卷积函数
    8.属性更新函数

成员变量：
    1.学习率
    2.训练次数
    3.W和b的形状
    4.输出文件夹
    5.输出文件名
    6.
'''

import os
import utils

from proc import Feed
from proc import Proc

import numpy as np
import tensorflow as tf

import platform

from termcolor import colored
from tensorflow.python.framework.ops import Tensor

class CNN:
    '''
    @about
        构造函数，从配置文件中读取所需信息，创建成员变量并赋值
    @param
        prop:   程序配置模块
        queue:  消息队列
    @return
        None
    '''
    def __init__(self,prop,name,queue=None):
        self.__logger = utils.getLogger()
        self.__sess = prop.queryAttr('session')

        self.__name = name
        self.__conv_size = prop.queryAttr('conv_size')
        self.__weights_shape = prop.queryAttr('weights_shape')

        # msg queue
        self.__queue = queue
        
        self.__features = self.__weights_shape[-1]
        self.__batch_c = prop.queryAttr('features') + prop.queryAttr('ifeatures')
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        self.__active_func = prop.queryAttr('active_func')
        self.__model_path = prop.queryAttr('model_path')
        self.__model_name = prop.queryAttr('model_name')
        self.__cols = prop.queryAttr('cols')
        self.__isTrain = True if prop.needTrain() else False
        self.__input1 = None
        self.__input2 = None
        if self.__isTrain:
            self.__batch_n = prop.queryAttr('batch_n')
            self.__loss_func = prop.queryAttr('loss_func')
            self.__learning_rate = prop.queryAttr('learning_rate')
            self.__max_round = prop.queryAttr('max_round')
            self.__data_dir = prop.queryAttr('data_dir')
            self.__optimizer = prop.queryAttr('optimizer')
        else:
            self.__batch_n = 1
        
                

    def __del__(self):
        self.__sess.close()


    ############################
    #########外部函数###########
    ############################

    '''
    @about
        初始化卷积网络：恢复或随机初始化
        默认先恢复，若失败再初始化
    @param
        ckpt:checkpoint路径
    @return
        True/False
    '''
    def init(self,ckpt):
        if not self.restore(ckpt):
            #self.__sess.run(tf.global_variables_initializer())
            self.__logger.debug('failed to restore,variables initialized!')
            return False
        else:
            self.__logger.debug('model restored')
            return True



    '''
    @about
        获取网络的输入值
    @param
        in1:输入图片
        in2:输入特征
    @return
        None
    '''
    def getInput(self,in1,in2):
        self.__input1 = in1
        self.__input2 = in2
        
        assert(self.__input1.shape[0] == self.__batch_n and self.__input2.shape[0] == self.__batch_n)
        #print(self.__input1.shape,self.__batch_h,self.__input2.shape)
        assert(self.__input1.shape[1] == self.__batch_h and self.__input2.shape[1] == self.__batch_h)
        assert(self.__input1.shape[2] == self.__batch_w and self.__input2.shape[2] == self.__batch_w)
        assert(self.__input1.shape[3] == self.__cols and self.__input2.shape[3] == self.__batch_c)

        #self.__input1 =
        #self.getHolder(self.__input1,[self.__batch_n,self.__batch_h,self.__batch_w,self.__cols],'input1')
        #self.__input2 =
        #self.getHolder(self.__input2,[self.__batch_n,self.__batch_h,self.__batch_w,self.__batch_c],'input2')
        
        w,b = self.getWeights(self.__conv_size,self.__weights_shape)
        self.__output = self.__input2
        for i in range(len(b)):
            self.__output = self.active(self.conv(self.__output,w[i]) + b[i],self.__active_func[i])

    '''
    @about
        使用卷积网络处理输入数据，并提供Proc抽象类，
        根据Proc实例处理数据，并返回处理结果
    @param
        proc:数据处理对象
        feed:数据输入对象
    @return
        返回卷积网络的输出经处理后的结果
        Tensor,[n,h,w,kxk]
    '''
    def process(self,proc,feed=None):
        # input1:image
        # input2:filters
        input1,input2 = proc.preprocess(self.__input1,self.__output)
        # some assertions
        assert(input1.shape[0] == input2.shape[0])
        assert(input1.shape[1] == input2.shape[1])
        assert(input1.shape[2] == input2.shape[2])
        assert(input1.shape[3] == self.__cols)
        #input1 [3,50,50,3]
        #input2 [1,50,50,121]
        #in1shape [3,50,50,3]
        #in2shape [3,50,50,8]

        predict = proc.process(input1,input2)

        '''
        # predict mode
        if not feed.isTrainMode():
            self.init()
            x1,x2,y = feed.getInputdata()
            predict = self.__sess.run(predict,feed_dict={self.__input1:x1,self.__input2:x2})
        '''
            
            #pred = np.reshape(predict,[-1,3])
            #print(pred[0:12])
            # debug-1 Z-Score
            #mean,vars = tf.nn.moments(self.__output,-1)
            #bn = mean.shape[0]
            #bh = mean.shape[1]
            #bw = mean.shape[2]
            #mean = tf.reshape(mean,[bn,bh,bw,1])
            #stds = tf.sqrt(vars)
            #stds = tf.reshape(stds,[bn,bh,bw,1])
            #eps = tf.constant(1e-2,tf.float32,[bn,bh,bw,1])
            #filters = (self.__output - mean) / (stds + eps)
            #filters = self.__sess.run(filters,feed_dict={self.__input2:x2})
            #filters = np.reshape(filters,[-1,self.__output.shape[3]])
            #print(filters.shape)
            #value = filters[0] * 255
            #print(np.sum(value))
            #means = np.mean(filters,-1)
            #filters = np.sum(filters,-1)
            #f = open('data/tmp','wb')
            #np.save(f,filters)
            #exit(-1)

            # debug-2 x-mean
            #mean,vars = tf.nn.moments(self.__output,-1)
            #bn = mean.shape[0]
            #bh = mean.shape[1]
            #bw = mean.shape[2]
            #mean = tf.reshape(mean,[bn,bh,bw,1])
            #diff = self.__output - mean
            #sum = tf.reduce_sum(diff ** 2,-1)
            #sum = tf.reshape(sum,[bn,bh,bw,1])
            #filters = (diff ** 2) / sum
            #filters = self.__sess.run(filters,feed_dict={self.__input2:x2})
            #f = open('data/tmp','wb')
            #np.save(f,filters)
            #f.close()
            #exit(-1)

            # debug-3 linear
            #mean,vars = tf.nn.moments(self.__output,-1)
            #bn = mean.shape[0]
            #bh = mean.shape[1]
            #bw = mean.shape[2]
            #mean = tf.reduce_mean(self.__output,-1)
            #mean = tf.reshape(mean,[bn,bh,bw,1])
            #sum = tf.reduce_sum(tf.abs(self.__output - mean),-1)
            #sum = tf.reshape(sum,[bn,bh,bw,1])

            #filters = (self.__output - mean) / sum
            #filters = self.__sess.run(filters,feed_dict={self.__input2:x2})
            #f = open('data/tmp','wb')
            #np.save(f,filters)
            #f.close()
            #exit(-1)

        predict = proc.postprocess(predict)

        return predict


    '''
    @about
        卷积网络用于自身训练的函数
    @param
        feed:Feed对象，调用其next_batch函数完成任务
        proc:Proc对象，调用其process函数完成任务
    @return
        None
    '''
    def train(self,feed,proc):
        self.__logger.debug('training...')
        assert(isinstance(proc,Proc))
        assert(isinstance(feed,Feed))
        assert(self.__Train)

        in1shape,in2shape = proc.getBatchShapes()

        # predict value
        predict = self.process(feed,proc)

        # ground truth
        truth = None
        truth = self.getHolder(truth,in1shape)

        #print(predict.shape,'=========',truth.shape)

        loss = self.loss(predict,truth,self.__loss_func)
        step = self.optimizer(loss,self.__learning_rate,self.__optimizer)

        # must be initialized when loss & step built.
        self.init()

        # max_round ==-1
        i = 0
        while self.__max_round < 0:
            assert(self.__max_round == -1)
            x1,x2,y = feed.next_batch()
            self.__sess.run(step,feed_dict={self.__input1:x1,self.__input2:x2,truth:y})

            xloss = self.__sess.run(loss,feed_dict={self.__input1:x1,self.__input2:x2,truth:y})
            self.__logger.debug('round:%d of inf,loss:%f...' % (i + 1,xloss))

            self.sendMsg([0,xloss])

            if i % 100 == 99:
                self.save(self.__model_path,self.__model_name)
            i += 1

        # max_round > 0
        for i in range(self.__max_round):
            x1,x2,y = feed.next_batch()
            self.__sess.run(step,feed_dict={self.__input1:x1,self.__input2:x2,truth:y})

            xloss = self.__sess.run(loss,feed_dict={self.__input1:x1,self.__input2:x2,truth:y})
            self.__logger.debug('round:%d of %d,loss:%f...' % (i + 1,self.__max_round, xloss))

            self.sendMsg([i / self.__max_round,xloss])

            if i % 100 == 99:
                self.save(self.__model_path,self.__model_name)
                

    '''
    @about
        向消息队列中发送数据
    @param
        msg:[current status,current loss]
    @return
        None
    '''
    def sendMsg(self,msg):
        if platform.system() != 'Windows':
            return
        try:
            self.__queue.put(msg)
        except:
            self.__logger.error('message queue is not specified.')

    '''
    @about
        以字符串输出所有信息
    @param
        None
    '''
    def toString(self):
        return 'cnn to string'


    ############################
    #########内部函数###########
    ############################

    '''
    @about
        卷积函数
    @param
        x:输入值
        w:卷积核
    '''
    def conv(self,x,W):
        return tf.nn.conv2d(x,W,[1,1,1,1],'SAME')

    '''
    @about
        获取卷积网络权重
    @param
        
    '''
    def getWeights(self,k,shape):
        ws = []
        bs = []
        for i in range(len(shape) - 1):
            w = tf.truncated_normal([k,k,shape[i],shape[i + 1]],dtype=tf.float32)
            w = tf.Variable(w)
            ws.append(w)
            b = tf.constant(0.1,tf.float32,[shape[i + 1]])
            b = tf.Variable(b)
            bs.append(b)
        return ws,bs


    '''
    @about
        获得占位符，此函数作为内部函数使用，可封闭整个CNN
    @param
        obj ：目标对象
        shape：目标形状
        name：名称
    @return
        返回占位符
    '''
    def getHolder(self,obj,shape,name=None):
        if name:
            obj = tf.placeholder(tf.float32,shape)
        else:
            obj = tf.placeholder(tf.float32,shape,name)
        return obj


    '''
    @about
        激活函数
    @param
        type
    '''
    def active(self,input,type):
        lst = ['relu','leaky_relu','softmax','sigmoid','tanh']
        assert(type in lst)
        if type == 'relu':
            return tf.nn.relu(input)
        elif type == 'sigmoid':
            return tf.nn.sigmoid(input)
        elif type == 'softmax':
            return tf.nn.softmax(input)
        elif type == 'tanh':
            return tf.nn.tanh(input)
        elif type == 'leaky_relu':
            return tf.nn.leaky_relu(input)
        else:
            assert(False)


    '''
    @about
        损失函数
    @param
        input:预测值
        output:目标值
        二者应相同维度
        type:损失函数类型
    @return
        损失值
    '''
    def loss(self,input,output,type='l1'):
        self.__logger.debug('computing loss...')
        lst = ['l1','cross_entropy']
        assert(type in lst)
        assert(isinstance(input,Tensor))
        assert(input.shape == output.shape)
        if type == 'l1':
            result = tf.reduce_mean(tf.abs(tf.subtract(input,output)))
        elif type == 'cross_entropy':
            result = tf.reduce_mean(-tf.reduce_sum(output * tf.log(input)))
        else:
            assert(False)
        self.__logger.debug('loss computed.')
        return result

    '''
    @about
        卷积网络优化器
    @param
        loss:损失值
        learning_rate:学习率
        type:优化器类型
    @return
        优化器
    '''
    def optimizer(self,loss,learning_rate,type):
        self.__logger.debug('building optimizer...')
        if type == 'Adam':
            result = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        elif type == 'Gradient':
            result = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        else:
            assert(False)
        self.__logger.debug('optimizer built.')
        return result

    '''
    @about
        存储
    @param
        path:文件路径
    @return
        True/False
    '''
    def save(self,path,name):
        self.__logger.info('saving model[%s]...' % self.__name)
        try:
            os.makedirs(path)
            self.__logger.info('directory \'%s\' was made for model saving.' % path)
        except OSError:
            if not os.path.isdir(path):
                self.__logger.error(colored('target path is not a directory','red'))
                return False
        saver = tf.train.Saver()
        saver.save(self.__sess, path + name)
        self.__logger.info('model[%s] saved.' % self.__name)
        return True

    '''
    @about
        恢复
    @param
        path:文件路径
    @return
        True/False
    '''
    def restore(self,path):
        if os.path.exists(path + '/checkpoint'):
            ckpt = tf.train.latest_checkpoint(path)
            self.__logger.debug('checkpoint path is \'%s\'' % ckpt)
            try:
                saver = tf.train.Saver(save_relative_paths=True)
                saver.restore(self.__sess, ckpt)
            except:
                self.__logger.warn(colored('model[%s] mismatch! current checkpoint will be overwriten,do you want to continue?' % self.__name,'yellow'))
                cands = ['y','yes','n','no']
                deci = ''
                while not(deci in cands):
                    deci = input(colored('y or n:','yellow'))
                    if deci == 'y' or deci == 'yes':
                        return False
                    elif deci == 'n' or deci == 'no':
                        exit(-1)
                    else:
                        self.__logger.warn(colored('invalid input,please try again...','yellow'))
                return False
            self.__logger.info('model restored from the latest checkpoint.')
        else:
            self.__logger.warn('checkpoint not found!')
            return False
        return True
