#!/bin/local/python3
# encoding:utf-8
'''
Filter,proc的子类，负责图像滤波
FilterFeed,feed的子类，与Filter配合使用，负责提供数据
'''
import os
import math
import utils
import numpy as np
import tensorflow as tf

from proc import Proc
from proc import Feed
from shared import Shared
from tensorflow.python.framework.ops import Tensor


class Filter(Proc):
    '''
    @about
        滤波对象初始化函数，确定各种参数
    @param
        prop:程序配置对象
    @return
        None
    '''

    def __init__(self, prop):
        sd = Shared()
        self.__logger = sd.getLogger()
        self.__isTrain = prop.needTrain()
        # 图像切片维度
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        self.__batch_n = prop.queryAttr('batch_n')
        # 空间域和图像域特征
        self.__features = prop.queryAttr('features')
        self.__ifeatures = prop.queryAttr('ifeatures')
        self.__batch_c = self.__features + self.__ifeatures
        self.__shape1 = [self.__batch_n, self.__batch_h, self.__batch_w, 3]
        self.__shape2 = [self.__batch_n, self.__batch_h, self.__batch_w, self.__batch_c]

    #########BEGIN###########
    ######可选的激活函数########
    #########################

    '''
    @about
        Z-Score,映射数据，使均值为0，方差为1
    @param
        input:输入数据
    @return
        激活后的数据，均值为0，方差为1
    '''

    def Z_Score(self, input):
        mean, vars = tf.nn.moments(input, -1)
        assert (len(mean.shape) == 3)

        bn = mean.shape[0]
        bh = mean.shape[1]
        bw = mean.shape[2]

        mean = tf.reshape(mean, [bn, bh, bw, 1])
        stds = tf.sqrt(vars)
        stds = tf.reshape(stds, [bn, bh, bw, 1])
        eps = tf.constant(1e-2, tf.float32, [bn, bh, bw, 1])
        output = (input - mean) / (stds + eps)
        return output

    '''
    @about
        映射数据至[0,1]之间，所有项之和为1
    @param
        input:输入数据
    @return
        激活后的数据
    '''

    def means(self, input):
        mean, vars = tf.nn.moments(input, -1)
        assert (len(mean.shape) == 3)

        bn = mean.shape[0]
        bh = mean.shape[1]
        bw = mean.shape[2]

        mean = tf.reshape(mean, [bn, bh, bw, 1])
        diff = input - mean
        sum = tf.reduce_sum(diff ** 2, -1)
        sum = tf.reshape(sum, [bn, bh, bw, 1])
        output = (diff ** 2) / sum
        return output

    '''
    @about
        线性比值激活函数
    @param
        input:输入数据
    @return
        激活后的数据
    '''

    def linear(self, input):
        mean, vars = tf.nn.moments(input, -1)
        assert (len(mean.shape) == 3)

        bn = mean.shape[0]
        bh = mean.shape[1]
        bw = mean.shape[2]

        mean = tf.reduce_mean(input, -1)
        mean = tf.reshape(mean, [bn, bh, bw, 1])
        sum = tf.reduce_sum(tf.abs(input - mean), -1)
        sum = tf.reshape(sum, [bn, bh, bw, 1])
        eps = tf.constant(1, tf.float32, [bn, bh, bw, 1])

        output = (input - mean) / (sum + eps)
        return output

    '''
    @about
        根据输入数据的类型进行归一化
    @param
        input:输入数据
        axis:默认为最后一维
    @return
        归一化后的数据
    '''

    def softmax(self, input, axis=-1):
        assert (len(input.shape) == 4)
        if isinstance(input, Tensor):
            result = tf.nn.softmax(input, axis)
        else:
            base = np.sum(np.exp(input), -1)
            result = np.exp(input) / np.reshape(base, [base.shape[0], base.shape[1], base.shape[2], 1])
        return result

    ###########END###########
    ######可选的激活函数########
    #########################

    '''
    @about
        预处理
        滤波核归一化
        确保图像维度正确
    @param
        img:        输入图像
        cnn_output: 卷积网络的输出结果
    @return
        img:        四维图像
        cnn_output: 滤波核
    '''

    def preprocess(self, img, cnn_output):
        bn = img.shape[0]
        bh = img.shape[1]
        bw = img.shape[2]

        # 数据激活
        cnn_mul = cnn_output * cnn_output
        cnn_sum = tf.reduce_sum(cnn_mul, -1)
        cnn_sum = tf.reshape(cnn_sum, [bn, bh, bw, 1])
        filters = cnn_mul / cnn_sum * 0.5

        assert (len(img.shape) == 4)
        return img, filters

    '''
    @about
        收尾工作，负责格式转换
    @param
        input:浮点型数组
    @return
        返回整数数组
    '''

    def postprocess(self, input):
        if isinstance(input, np.ndarray):
            input = input.astype(np.uint8)

        return input

    '''
    @about
        滤波函数
        输入图像与其对应的滤波器
        先将图像进行填充，然后滤波
        输出滤波后的图像
    @param
        img与filters 前三个维度应相同
        img[np.ndarray]:输入的图像[n,h,w,c]
        filters[placeholder]:图像对应的滤波核[n,h,w,kxk]，各层应归一化
    @return
        返回滤波后的图像[n,h,w,c]
    '''

    def process(self, img, filters):
        assert (isinstance(filters, Tensor) or isinstance(filters, np.ndarray))

        img = np.reshape(img, [-1, img.shape[0], img.shape[1], img.shape[2]]) if len(img.shape) == 3 else img

        assert (img.shape[0] == filters.shape[0])
        assert (img.shape[1] == filters.shape[1])
        assert (img.shape[2] == filters.shape[2])

        # 图像维度
        ih = img.shape[1]
        iw = img.shape[2]
        ic = 3

        # 滤波核维度
        bn = filters.shape[0]
        bh = filters.shape[1]
        bw = filters.shape[2]
        bc = filters.shape[3]
        bk = int(math.sqrt(int(filters.shape[3])))

        # 镜像填充
        img_padded = self.padding(img, bk)

        self.__logger.debug('filtering images...')

        # 滤波
        result = None
        for n in range(bn):
            tmp = tf.constant(0, tf.float32, [1, ih, iw, ic])
            for i in range(bk * bk):
                sx = int(i % bk)
                sy = int(i / bk)
                blk = tf.slice(img_padded, [n, sy, sx, 0], [1, ih, iw, ic])
                flt = tf.slice(filters, [n, 0, 0, i], [1, ih, iw, 1])
                tmp = tf.concat([blk * flt, tmp], 0)
            tmp = tf.reduce_sum(tmp, 0)
            assert (tmp.shape == [ih, iw, ic])
            tmp = tf.reshape(tmp, [1, ih, iw, ic])
            try:
                result = tf.concat([result, tmp], 0)
            except:
                result = tmp

        self.__logger.debug('images filtered.')

        return result

    '''
    @about
        以对称方式填充图像，用于滤波前
        numpy方式的实现速度更快，但不能处理Tensor数据
    @param
        img[np.ndarray]:待处理图片[n,h,w,c]
        k:  滤波核大小
    @return
        返回[n,h+k-1,w+k-1,c]大小的填充图
    '''

    def padding(self, img, k):
        r = int(k / 2)
        if isinstance(img, Tensor):
            assert (len(img.shape) == 4)
            result = tf.pad(img, [[0, 0], [r, r], [r, r], [0, 0]], mode='SYMMETRIC')
        else:
            assert (isinstance(img, np.ndarray))
            n = img.shape[0]
            h = img.shape[1]
            w = img.shape[2]
            c = img.shape[3]
            result = np.zeros([n, h + k - 1, w + k - 1, c])
            result[:, r:h + r, r:w + r, :] = img
            result[:, r:h + r, :r, :] = result[:, r:h + r, r:r + r, :][:, :, ::-1, :]
            result[:, h + r:h + r + r, :w + r + r, :] = result[:, h:h + r, :w + r + r, :][:, ::-1, :, :]
            result[:, r:h + r + r, w + r:w + r + r, :] = result[:, r:h + r + r, w:w + r, :][:, :, ::-1, :]
            result[:, :r, :, :] = result[:, r:r + r, :, :][:, ::-1, :, :]

        return result

    '''
    @about
        根据src类型进行单层切片
    @param
        1.
        src[np.ndarray]:[h,w,c]
        begin:[y,x]
        2.src[Tensor]:[n,h,w,c]
        begin:[i,y,x,0]
        size:切片大小
    @return
        返回[size*size,3]的矩阵
    '''

    def getBlock(self, src, begin, size, shape):
        if isinstance(src, Tensor):
            assert (len(begin) == 4)
            block = tf.slice(src, begin, [1, size, size, src.shape[3]])
            block = tf.reshape(block, shape)
        else:
            assert (isinstance(src, np.ndarray))
            assert (len(begin) == 2)
            block = src[begin[0]:begin[0] + size, begin[1]:begin[1] + size, :]
            block = np.reshape(block, shape)
        return block

    '''
    @about
        返回处理单元的shape
    @param
        None
    @return
        [n,h,w,c]
    '''

    def getBatchShapes(self):
        assert (self.__isTrain)
        return self.__shape1, self.__shape2


'''
与Filter对象配合使用，为其提供数据
'''


class FilterFeed(Feed):
    '''
    @about
        构造函数，初始化参数
    @param
        prop:程序配置对象
    @return
        None
    '''

    def __init__(self, prop):
        sd = Shared()
        self.__logger = sd.getLogger()
        self.__isTrain = True if prop.needTrain() else False
        self.__input_dir = prop.queryAttr('data_dir')
        self.__scene_name = os.listdir(self.__input_dir)
        self.__cols = prop.queryAttr('cols')
        if self.__isTrain:
            self.__directories = [(self.__input_dir + itm + '/') for itm in self.__scene_name]
        self.__ground_truth = prop.queryAttr('ground_truth')
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        self.__batch_n = prop.queryAttr('batch_n')
        self.__features = prop.queryAttr('features')
        self.__ifeatures = prop.queryAttr('ifeatures')
        self.__batch_c = self.__features + self.__ifeatures
        self.__cache_size = prop.queryAttr('cache_size')
        self.__cache = prop.queryAttr('cache')
        self.__abandons = []

    '''
    @about
        指定feed的文件读取参数
    @param
        img_name:图像路径
        txt_name:文本路径
    @return
        None
    '''

    def setInputName(self, img_name, txt_name, conf_name):
        self.__txt_input = txt_name
        self.__img_input = img_name
        self.__conf_input = conf_name

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
        assert (not self.__isTrain)
        features = self.__features
        ifeatures = self.__ifeatures
        # check md5
        imgpath = self.__input_dir + self.__img_input
        txtpath = self.__input_dir + self.__txt_input
        confpath = self.__input_dir + self.__conf_input
        isMatch = self.checkMatch(imgpath, txtpath, confpath)
        if not isMatch:
            self.__logger.error('md5 of scene [%s] mismatch,exit(-1)' % self.__input_dir)
            exit(-1)
        try:
            # value = self.__cache.get(self.__input_dir)
            a = self.__cache.get(self.__input_dir + self.__img_input)
            b = self.__cache.get(self.__input_dir + self.__txt_input)
            c = self.__cache.get(self.__input_dir + self.__ground_truth)
            value = {'input': a, 'data': b, 'truth': c}
        except:
            a = utils.readIMG(self.__input_dir + self.__img_input)
            b = utils.readTXT(self.__input_dir + self.__txt_input)
            c = utils.readIMG(self.__input_dir + self.__ground_truth)
            isMatch = self.checkMatch(imgpath, txtpath, confpath)
            self.__cache.add(self.__input_dir + self.__img_input, a)
            self.__cache.add(self.__input_dir + self.__txt_input, b)
            self.__cache.add(self.__input_dir + self.__ground_truth, c)
            value = {'input': a, 'data': b, 'truth': c}
            # some assertions about input data
            assert (len(value['input'].shape) == 4)
            assert (len(value['truth'].shape) == 4)
            assert (self.__batch_h <= value['input'].shape[1])
            assert (self.__batch_w <= value['input'].shape[2])
            assert (value['input'].shape[1] == value['truth'].shape[1])
            assert (value['input'].shape[2] == value['truth'].shape[2])
            # height & width of the input image
            ih = value['input'].shape[1]
            iw = value['input'].shape[2]
            value['data'] = np.reshape(value['data'], [-1, ih, iw, features])
            # x1 Image,[n,h,w,3]

            # sx = sy = 500
            sx = sy = 400

            bh = self.__batch_h
            bw = self.__batch_w
            bc = self.__batch_c
            cols = self.__cols

            x1 = value['input'][:, sy:sy + bh, sx:sx + bw, :]
            y = value['truth'][:, sy:sy + bh, sx:sx + bw, :]

            # x2 text,[n,h,w,c]
            x2 = np.zeros([value['data'].shape[0], bh, bw, bc])
            # add features from txt
            x2[:, :, :, :value['data'].shape[3]] = value['data'][:, sy:sy + bh, sx:sx + bw, :]
            # select a layer of txt randomly
            idx = np.random.randint(0, x2.shape[0])

            idx = 0

            x2 = x2[idx:idx + 1, :, :, :]
            # add features subtracted from input image
            assert (ifeatures == 2)
            x2[:, :, :, features + 0:features + 1] = utils.getLuminance(x1)
            x2[:, :, :, features + 1:features + 2] = utils.getMagnitude(x1)
            # x2[:,:,:,features + 2:features + 5] = np.reshape(x1,[bh,bw,cols])
            # normalize x2
            x2 = self.normalize(x2)
        return x1, x2, y

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

    def checkMatch(self, imgpath, txtpath, confpath):
        # return True
        in1md5 = utils.md5sum(imgpath)
        in2md5 = utils.md5sum(txtpath)
        ta1md5 = utils.getJsonAttr('ImageMD5', None, confpath)
        ta2md5 = utils.getJsonAttr('TextMD5', None, confpath)
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
        return [bn, bh, bw, cols], [bn, bh, bw, bc]

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

    def normalize(self, input):
        assert (isinstance(input, np.ndarray))
        # Z-Score
        # result = (input - np.mean(input)) / np.std(input)
        # Sigmoid
        # result = 1 / (1 + np.exp(-input))
        # tanh
        result = np.tanh(input)
        return result

    '''
    @about
        内部函数
        获取输入的图像与其对应的目标图，用于训练
        为提升速度，每次只返回一个场景的多个切片
    @param
        seed：随机数种子
    @return
        返回n图和对应的n特征
        x1:[n,h,w,3]
        x2:[n,h,w,c]
        y :[n,h,w,3]
    '''

    def next_batch(self, seed):
        np.random.seed(seed)
        assert (self.__isTrain)
        self.__logger.debug('Size of Cache:%dM' % self.__cache.getSize())
        self.__logger.debug('Eta of Cache:%.2f%%' % (self.__cache.getEta() * 100.0))
        cols = self.__cols
        bn = self.__batch_n
        bh = self.__batch_h
        bw = self.__batch_w
        bc = self.__batch_c
        features = self.__features
        # x1 input image,x2 txt features,y ground truth
        x1 = np.zeros([bn, bh, bw, cols])
        x2 = np.zeros([bn, bh, bw, bc])
        y = np.zeros([bn, bh, bw, cols])
        # select a scene randomly
        id = np.random.randint(0, len(self.__directories))
        while self.__directories[id] in self.__abandons:
            id = np.random.randint(0, len(self.__directories))
            self.__logger.debug('something is wrong in this scene,looking for another one...')
        # iter each layer of batchs
        for n in range(bn):
            # get input data
            a = b = c = None
            imgpath = self.__directories[id] + self.__img_input
            txtpath = self.__directories[id] + self.__txt_input
            confpath = self.__directories[id] + self.__conf_input
            # exit(-1)
            # print(imgpath,txtpath,confpath)
            isMatch = self.checkMatch(imgpath, txtpath, confpath)
            # checkMatch before putting them into Cache
            # skip current scene when md5 mismatch
            if not isMatch:
                n -= 1
                self.__abandons.append(self.__directories[id])
                self.__logger.warn('md5 of scene [%s] mismatch,abandon this one!' % self.__directories[id])
                continue
            try:
                a = self.__cache.get(self.__directories[id] + self.__img_input)
            except:
                a = utils.readIMG(self.__directories[id] + self.__img_input)
                self.__cache.add(self.__directories[id] + self.__img_input, a)
            try:
                b = self.__cache.get(self.__directories[id] + self.__txt_input)
            except:
                b = utils.readTXT(self.__directories[id] + self.__txt_input)
                self.__cache.add(self.__directories[id] + self.__txt_input, b)
            try:
                c = self.__cache.get(self.__directories[id] + self.__ground_truth)
            except:
                c = utils.readIMG(self.__directories[id] + self.__ground_truth)
                self.__cache.add(self.__directories[id] + self.__ground_truth, c)

            value = {'input': a, 'data': b, 'truth': c}
            # some assertions about input data
            assert (len(value['input'].shape) == 4)
            assert (len(value['truth'].shape) == 4)
            assert (value['input'].shape[1] == value['truth'].shape[1])
            assert (value['input'].shape[2] == value['truth'].shape[2])
            # height & width of the input image
            ih = value['input'].shape[1]
            iw = value['input'].shape[2]
            value['data'] = np.reshape(value['data'], [-1, ih, iw, features])
            # pick a sample pos(left&up corner) randomly
            sh = np.random.randint(0, ih - bh)
            sw = np.random.randint(0, iw - bw)
            sh = sw = 400
            # part of input image(x1) & ground truth(y)
            px1 = utils.slice(value['input'], [0, sh, sw, 0], [1, bh, bw, cols])
            py = utils.slice(value['truth'], [0, sh, sw, 0], [1, bh, bw, cols])
            # select a layer of txt randomly
            idx = np.random.randint(0, value['data'].shape[0])

            # self.__logger.debug('%d %d %d %d' % (id,idx,sh,sw))

            # idx = 0
            # part of input txt(px2)
            px2 = np.zeros([1, bh, bw, bc])
            # [ 0:features]:features stored in txt
            px2[:, :, :, :features] = utils.slice(value['data'], [idx, sh, sw, 0], [1, bh, bw, features])
            # [features:]:features extracted from img
            assert (self.__ifeatures == 2)
            px2[:, :, :, features + 0:features + 1] = utils.getLuminance(px1)
            px2[:, :, :, features + 1:features + 2] = utils.getMagnitude(px1)
            # px2[:,:,:,features + 2:features + 5] =
            # np.reshape(px1,[bh,bw,cols])
            px2 = np.reshape(px2, [-1, bc])
            # reshape these parts
            x1[n] = np.reshape(px1, [bh, bw, cols])
            x2[n] = np.reshape(px2, [bh, bw, bc])
            y[n] = np.reshape(py, [bh, bw, cols])
        # normalize data before return them
        x2 = self.normalize(x2)
        # utils.displayImage(x1)
        return x1, x2, y
