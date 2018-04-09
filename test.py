#!/bin/local/python3
#encoding:utf-8
'''
CNN测试模块
用于测试卷积网络，检查各环节的正确性
'''

import os
import math
import utils

from proc import Proc
from proc import Feed
from cache import Cache

import numpy as np
import tensorflow as tf

from tensorflow.python.framework.ops import Tensor

class Test(Proc):
    
    def __init__(self,prop):
        self.__prop = prop
        return

    def preprocess(self, input1, input2):
        mean,vars = tf.nn.moments(input2,-1)
        assert(len(mean.shape) == 3)

        bn = mean.shape[0]
        bh = mean.shape[1]
        bw = mean.shape[2]

        mean = tf.reshape(mean,[bn,bh,bw,1])
        stds = tf.sqrt(vars)
        stds = tf.reshape(stds,[bn,bh,bw,1])

        filters = (input2 - mean) / stds
        # filters = self.softmax(cnn_output)
        # filters = cnn_output
        return input1,filters

    '''
    @about
        处理函数
    @param
        input1:模型输入
        input2:CNN输出
    @return
        val:程序返回值
    '''
    def process(self, input1, input2):
        # val = self.softmax(input2)
        # return input1 * input2
        return input2
        # return val
        # return super().process(input1, input2)

    def postprocess(self):
        return

    def getBatchShapes(self):
        return TestFeed(self.__prop).getBatchShapes()

    def softmax(self,input):
        assert(len(input.shape) == 4)
        if isinstance(input,Tensor):
            result = tf.nn.softmax(input)
        else:
            base = np.sum(np.exp(input),-1)
            #print(input.shape,input)
            #print('=====')
            #print(base.shape,base)
            #print('exit')
            #exit(-1)
            result = np.exp(input) / np.reshape(base,[base.shape[0],base.shape[1],base.shape[2],1])
        return result
        

class TestFeed(Feed):
    def __init__(self,prop):
        self.__logger = utils.getLogger()
        self.__Train = True if prop.needTrain() else False
        self.__input_dir = prop.queryAttr('data_dir')
        self.__scene_name = os.listdir(self.__input_dir)
        self.__cols = prop.queryAttr('cols')
        if self.__Train:
            self.__directories = [(self.__input_dir + itm + '/') for itm in self.__scene_name]
        self.__txt_input = prop.queryAttr('txt_input')
        self.__img_input = prop.queryAttr('img_input')
        self.__conf_input = prop.queryAttr('conf_input')
        self.__img_output = prop.queryAttr('img_output')
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        self.__batch_n = prop.queryAttr('batch_n')
        self.__features = prop.queryAttr('features')
        self.__ifeatures = prop.queryAttr('ifeatures')
        self.__batch_c = self.__features + self.__ifeatures
        self.__cache_size = prop.queryAttr('cache_size')
        # properties specified in config.json
        # Cache that store scenes
        self.__cache = Cache(self.__cache_size)
        # store scene names that don't match
        self.__abandons = []

        # temp
        self.__rx1 = utils.readIMG('d:/downloads/data/raw.png')
        self.__rx2 = self.__rx1
        self.__ry = utils.readIMG('d:/downloads/data/truth.png')

    '''
    @about
        析构函数
    @param
        None
    @return
        None
    '''
    def __del__(self):
        self.__logger.debug('Eta of Cache:%.2f%%' % (self.__cache.getEta() * 100.0))


    '''
    @about
        外部函数
        获取输入的图像，用于预测
    @param
        None
    @return
        np.ndarray 或 Tensor
        x1:img   [n,h,w,3]
        x2:txt   [n,h,w,c]
    '''
    def getInputdata(self):
        if self.__Train:
            assert(False)
        else:
            features = self.__features
            ifeatures = self.__ifeatures            
            # check md5
            imgpath = self.__input_dir + self.__img_input
            txtpath = self.__input_dir + self.__txt_input
            confpath = self.__input_dir + self.__conf_input
            isMatch = self.checkMatch(imgpath,txtpath,confpath)
            if not isMatch:
                self.__logger.error('md5 of scene [%s] mismatch,exit(-1)' % self.__input_dir)
                exit(-1)
            try:
                value = self.__cache.get(self.__input_dir)
            except:
                a = utils.readIMG(self.__input_dir + self.__img_input)
                b = utils.readTXT(self.__input_dir + self.__txt_input)
                c = utils.readIMG(self.__input_dir + self.__img_output)
                imgpath = self.__input_dir + self.__img_input
                txtpath = self.__input_dir + self.__txt_input
                confpath = self.__input_dir + self.__conf_input
                isMatch = self.checkMatch(imgpath,txtpath,confpath)
                # skip current scene when md5 mismatch
                if not isMatch:
                    self.__abandons.append(self.__directories[id])
                    self.__logger.error('md5 of scene [%s] mismatch,abandon this one!' % self.__directories[id])
                    return self.getInputdata()
                # checkMatch before putting them into Cache
                self.__cache.add(self.__input_dir,{'input':a,'data':b,'truth':c})
                value = self.__cache.get(self.__input_dir)
            # some assertions about input data
            assert(len(value['input'].shape) == 4)
            assert(len(value['truth'].shape) == 4)
            assert(self.__batch_h <= value['input'].shape[1])
            assert(self.__batch_w <= value['input'].shape[2])
            assert(value['input'].shape[1] == value['truth'].shape[1])
            assert(value['input'].shape[2] == value['truth'].shape[2])
            # height & width of the input image
            ih = value['input'].shape[1]
            iw = value['input'].shape[2]
            value['data'] = np.reshape(value['data'],[-1,ih,iw,features])
            # x1 Image,[n,h,w,3]

            t = 0

            x1 = value['input'][:,:self.__batch_h,t:t + self.__batch_w,:]
            
            y = value['truth'][:,:self.__batch_h,:self.__batch_w,:]

            # x2 text,[n,h,w,c]
            x2 = np.zeros([value['data'].shape[0],self.__batch_h,self.__batch_w,self.__batch_c])
            # add features from txt
            x2[:,:,:,:value['data'].shape[3]] = value['data'][:,:self.__batch_h,t:t + self.__batch_w,:]
            # select a layer of txt randomly
            idx = np.random.randint(0,x2.shape[0])
            x2 = x2[idx:idx + 1,:,:,:]
            # add features subtracted from input image
            assert(ifeatures == 2)
            x2[:,:,:,features + 0:features + 1] = utils.getLuminance(x1)
            x2[:,:,:,features + 1:features + 2] = utils.getMagnitude(x1)
            # normalize x2
            x2 = self.normalize(x2)
        return x1,x2,y


    '''
    @about
        外部函数，以字符串形式输出属性信息
    @param
        None
    '''
    def toString(self):
        result = '\n\n\t\t[Feed]:\n'
        return result

    '''
    @about
        通过校验文件MD5，检查三者是否匹配
    @param
        in1:图像文件路径
        in2:文本文件路径
        in3:配置文件路径
    @return
        True/False
    '''
    def checkMatch(self,imgpath,txtpath,confpath):
        return True
        # skip check
        in1md5 = utils.md5sum(imgpath)
        in2md5 = utils.md5sum(txtpath)
        scene_name = re.split(r'\.',imgpath)[-1]
        ta1md5 = utils.getJsonAttr('ImageMD5',None,confpath)
        ta2md5 = utils.getJsonAttr('TextMD5',None,confpath)
        return in1md5 == ta1md5 and in2md5 == ta2md5


    '''
    @about
        返回输入值形状
    @param
        None
    @return
        img.shape,txt.shape
    '''
    def getBatchShapes(self):
        cols = self.__cols
        bn = self.__batch_n
        bh = self.__batch_h
        bw = self.__batch_w
        bc = self.__batch_c
        return [bn,bh,bw,cols],[bn,bh,bw,bc]

    '''
    @about
        对输入数据进行归一化处理
        采用Z-Score方式或Sigmoid方式
        默认采用Sigmoid
    @param
        input:输入数据[np.ndarray]
    @return
        对指定轴进行标准化后的数据
    '''
    def normalize(self,input):
        assert(isinstance(input,np.ndarray))
        # Z-Score
        # result = (input - np.mean(input)) / np.std(input)
        # Sigmoid
        result = 1 / (1 + np.exp(-input))
        return result
    

    '''
    @about
        返回当前是否为训练模式
    @param
        None
    @return
        True/False
    '''
    def isTrainMode(self):
        return self.__Train

    '''
    @about
        内部函数
        获取输入的图像与其对应的目标图，用于训练
        为提升速度，每次只返回一个场景的多个切片
    @param
        None
    @return
        返回n图和对应的n特征
        x1:[n,h,w,3]
        x2:[n,h,w,c]
        y :[n,h,w,3]
    '''
    def next_batch(self):
        '''
        x1 = self.__rx1
        x2 = self.__rx2
        y = self.__ry
        return x1,x2,y
        '''
        ##
        assert(self.__Train)
        self.__logger.debug('Size of Cache:%dM' % self.__cache.getSize())
        self.__logger.debug('Eta of Cache:%.2f%%' % (self.__cache.getEta() * 100.0))
        cols = self.__cols
        bn = self.__batch_n
        bh = self.__batch_h
        bw = self.__batch_w
        bc = self.__batch_c
        features = self.__features
        # x1 input image,x2 txt features,y ground truth
        x1 = np.zeros([bn,bh,bw,cols])
        x2 = np.zeros([bn,bh,bw,bc])
        y = np.zeros([bn,bh,bw,cols])
        # select a scene randomly
        id = np.random.randint(0,len(self.__directories))
        while self.__directories[id] in self.__abandons:
            id = np.random.randint(0,len(self.__directories))
            self.__logger.debug('refresh the value of id...')
        # iter each layer of batchs
        for n in range(bn):
            # get input data
            try:
                value = self.__cache.get(self.__directories[id])
            except:
                a = utils.readIMG(self.__directories[id] + self.__img_input)
                b = utils.readTXT(self.__directories[id] + self.__txt_input)
                c = utils.readIMG(self.__directories[id] + self.__img_output)
                imgpath = self.__directories[id] + self.__img_input
                txtpath = self.__directories[id] + self.__txt_input
                confpath = self.__directories[id] + self.__conf_input
                isMatch = self.checkMatch(imgpath,txtpath,confpath)
                # skip current scene when md5 mismatch
                if not isMatch:
                    n-=1
                    self.__abandons.append(self.__directories[id])
                    self.__logger.error('md5 of scene [%s] mismatch,abandon this one!' % self.__directories[id])
                    continue
                # checkMatch before putting them into Cache
                self.__cache.add(self.__directories[id],{'input':a,'data':b,'truth':c})
                value = self.__cache.get(self.__directories[id])
            # some assertions about input data
            assert(len(value['input'].shape) == 4)
            assert(len(value['truth'].shape) == 4)
            assert(value['input'].shape[1] == value['truth'].shape[1])
            assert(value['input'].shape[2] == value['truth'].shape[2])
            # height & width of the input image
            ih = value['input'].shape[1]
            iw = value['input'].shape[2]
            value['data'] = np.reshape(value['data'],[-1,ih,iw,features])
            # pick a sample pos(left&up corner) randomly
            sh = np.random.randint(0,ih - bh + 1)
            sw = np.random.randint(0,iw - bw + 1)
            sh = sw = 0
            # part of input image(x1) & ground truth(y)
            px1 = utils.slice(value['input'],[0,sh,sw,0],[1,bh,bw,cols])
            py = utils.slice(value['truth'],[0,sh,sw,0],[1,bh,bw,cols])
            # select a layer of txt randomly
            idx = np.random.randint(0,value['data'].shape[0])
            idx = 0
            # part of input txt(px2)
            px2 = np.zeros([1,bh,bw,bc])
            # [ 0:features]:features stored in txt
            px2[:,:,:,:features] = utils.slice(value['data'],[idx,sh,sw,0],[1,bh,bw,features])
            # [features:]:features extracted from img
            assert(self.__ifeatures == 2)
            px2[:,:,:,features + 0:features + 1] = utils.getLuminance(px1)
            px2[:,:,:,features + 1:features + 2] = utils.getMagnitude(px1)
            px2 = np.reshape(px2,[-1,bc])
            # reshape these parts
            x1[n] = np.reshape(px1,[bh,bw,cols])
            x2[n] = np.reshape(px2,[bh,bw,bc])
            y[n] = np.reshape(py,[bh,bw,cols])
        # normalize data before return them
        x2 = self.normalize(x2)
        return x1,x2,y
