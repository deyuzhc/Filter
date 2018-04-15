#!/bin/local/python3
# encoding:utf-8
'''
根据模型配置对象(feed)构建模型
输入[prop,feed]
输出[output]

外部接口：
    1.初始化
    2.建立网络流
    3.输入、输出

内部函数：
    1.获取属性值
    2.获取W与b
    3.初始化参数
    4.多种激活函数
    5.多种损失函数
    6.多种优化器
    7.卷积函数
'''

import os
import utils
import colorama
import platform
import tensorflow as tf

from proc import Feed
from proc import Proc
from termcolor import cprint
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

    def __init__(self, prop, name, msg_queue=None):
        # 全局唯一日志句柄
        self.__logger = utils.getLogger()
        # 全局唯一会话
        self.__sess = prop.queryAttr('session')
        # self.__sess = utils.getSession()
        # 卷积网络的字符串命名
        self.__name = name
        # 消息队列，用于中转训练误差数据
        self.__queue = msg_queue
        # 空间域特征
        features = prop.queryAttr('features')
        # 图像域特征
        ifeatures = prop.queryAttr('ifeatures')
        # 输入切片的维度
        self.__batch_c = features + ifeatures
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        # 图像通道，一般为3
        self.__cols = prop.queryAttr('cols')
        # 卷积网络模型参数
        self.__conv_size = prop.queryAttr('conv_size')
        self.__model_path = prop.queryAttr('model_path')
        self.__ckpt_name = prop.queryAttr('ckpt_name')
        self.__active_func = prop.queryAttr('active_func')
        self.__weights_shape = prop.queryAttr('weights_shape')
        self.__isTrain = True if prop.needTrain() else False
        self.__input1 = None
        self.__input2 = None
        if self.__isTrain:
            self.__batch_n = prop.queryAttr('batch_n')
            self.__loss_func = prop.queryAttr('loss_func')
            self.__learning_rate = prop.queryAttr('learning_rate')
            self.__max_round = prop.queryAttr('max_round')
            # 输入数据目录
            self.__data_dir = prop.queryAttr('data_dir')
            self.__optimizer = prop.queryAttr('optimizer')
        else:
            self.__batch_n = 1
        colorama.init()

    '''
    @about
        析构函数，负责关闭会话
        内存占用由回收机制负责
    @param
        None
    @return
        None
    '''

    def __del__(self):
        self.__sess.close()

    ############################
    #########外部接口###########
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

    def init(self, ckpt):
        if not self.restore(ckpt):
            self.__sess.run(tf.global_variables_initializer())
            self.__logger.debug('failed to restore,variables initialized!')
        else:
            self.__logger.debug('model restored')

    '''
    @about
        建立卷积网络数据流
        将数据前向输入到卷积网络中
    @param
        in1:输入图片
        in2:输入特征
    @return
        None
    '''

    def build(self, in1, in2):
        self.__input1 = in1
        self.__input2 = in2

        assert (self.__input1.shape[0] == self.__batch_n and self.__input2.shape[0] == self.__batch_n)
        assert (self.__input1.shape[1] == self.__batch_h and self.__input2.shape[1] == self.__batch_h)
        assert (self.__input1.shape[2] == self.__batch_w and self.__input2.shape[2] == self.__batch_w)
        assert (self.__input1.shape[3] == self.__cols and self.__input2.shape[3] == self.__batch_c)

        w, b = self.getWeights(self.__conv_size, self.__weights_shape)
        self.__output = self.__input2
        for i in range(len(b)):
            self.__output = self.active(self.conv(self.__output, w[i]) + b[i], self.__active_func[i])

    '''
    @about
        使用卷积网络处理输入数据
        根据Proc实例处理数据，并返回处理结果
    @param
        proc:数据处理对象
    @return
        返回卷积网络的输出经处理后的结果
        Tensor,[n,h,w,kxk]
    '''

    def process(self, proc):
        # input1:图像;input2:滤波核
        input1, input2 = proc.preprocess(self.__input1, self.__output)
        # 保证两输入维度正确
        assert (input1.shape[0] == input2.shape[0])
        assert (input1.shape[1] == input2.shape[1])
        assert (input1.shape[2] == input2.shape[2])
        assert (input1.shape[3] == self.__cols)

        predict = proc.process(input1, input2)

        predict = proc.postprocess(predict)

        return predict

    '''
    @about
        向消息队列中发送数据
    @param
        msg:[current status,current loss]
    @return
        None
    '''

    def sendMsg(self, msg):
        if platform.system() != 'Windows':
            return
        try:
            self.__queue.put(msg)
        except:
            self.__logger.error('message queue is not specified.')

    '''
    @about
        存储
    @param
        path:文件路径
    @return
        True/False
    '''

    def save(self, path, name, save_round):
        utils.setFlag('safeExit', False)
        self.__logger.info('saving model[%s]...' % self.__name)
        try:
            os.makedirs(path)
            self.__logger.info('directory \'%s\' was made for model saving.' % path)
        except OSError:
            if not os.path.isdir(path):
                self.__logger.error(cprint('target path is not a directory', 'red'))
                return False
        saver = tf.train.Saver()
        saver.save(self.__sess, path + name)
        try:
            meta = utils.readJson(path + + 'meta.json')
        except:
            meta = {}
            meta['name'] = name
            meta['weights'] = 'tmp'
            meta['active_func'] = 'tmp'
            meta['round'] = save_round

        meta['round'] += save_round
        utils.writeJson(meta, path + 'meta.json')
        self.__logger.info('model[%s] saved.' % self.__name)
        utils.setFlag('safeExit', True)
        return True

    '''
    @about
        恢复
    @param
        path:文件路径
    @return
        True/False
    '''

    def restore(self, path):
        if os.path.exists(path + '/checkpoint'):
            ckpt = tf.train.latest_checkpoint(path)
            self.__logger.debug('checkpoint path is \'%s\'' % ckpt)
            try:
                saver = tf.train.Saver(save_relative_paths=True)
                saver.restore(self.__sess, ckpt)
            except:
                cprint(
                    'model[%s] mismatch! current checkpoint will be overwriten,do you want to continue?' % self.__name,
                    'yellow')
                cands = ['y', 'yes', 'n', 'no']
                deci = ''
                while not (deci in cands):
                    deci = input(cprint('y or n:', 'yellow'))
                    if deci == 'y' or deci == 'yes':
                        return False
                    elif deci == 'n' or deci == 'no':
                        exit(-1)
                    else:
                        self.__logger.warn('invalid input,please try again...')
                return False
            self.__logger.info('model restored from the latest checkpoint.')
        else:
            self.__logger.warn('checkpoint not found!')
            return False
        return True

    ############################
    ##########内部函数###########
    #######内部调用或调试使用######
    ############################

    '''
    @about
        卷积网络用于自身训练的函数
    @param
        feed:Feed对象，调用其next_batch函数完成任务
        proc:Proc对象，调用其process函数完成任务
    @return
        None
    '''

    def train(self, feed, proc):
        self.__logger.debug('training...')
        assert (isinstance(proc, Proc))
        assert (isinstance(feed, Feed))
        assert (self.__isTrain)

        in1shape, in2shape = proc.getBatchShapes()

        predict = self.process(feed, proc)

        truth = None
        truth = self.getHolder(truth, in1shape)

        loss = self.loss(predict, truth, self.__loss_func)
        step = self.optimizer(loss, self.__learning_rate, self.__optimizer)

        self.init()

        i = 0
        # self.__max_round < 0
        while self.__max_round < 0:
            assert (self.__max_round == -1)
            x1, x2, y = feed.next_batch()
            self.__sess.run(step, feed_dict={self.__input1: x1, self.__input2: x2, truth: y})

            xloss = self.__sess.run(loss, feed_dict={self.__input1: x1, self.__input2: x2, truth: y})
            self.__logger.debug('round:%d of inf,loss:%f...' % (i + 1, xloss))

            self.sendMsg([0, xloss])

            if i % 100 == 99:
                self.save(self.__model_path, self.__ckpt_name)
            i += 1

        # max_round > 0
        for i in range(self.__max_round):
            x1, x2, y = feed.next_batch()
            self.__sess.run(step, feed_dict={self.__input1: x1, self.__input2: x2, truth: y})

            xloss = self.__sess.run(loss, feed_dict={self.__input1: x1, self.__input2: x2, truth: y})
            self.__logger.debug('round:%d of %d,loss:%f...' % (i + 1, self.__max_round, xloss))

            self.sendMsg([i / self.__max_round, xloss])

            if i % 100 == 99:
                self.save(self.__model_path, self.__ckpt_name)

    '''
    @about
        卷积函数，输入数据与卷积核大小，进行卷积
    @param
        x:输入值
        w:卷积核
    @return
        用W对x做卷积之后的结果
    '''

    def conv(self, x, W):
        return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')

    '''
    @about
        根据指定参数，生成卷积网络权重
    @param
        k:各层之间的卷积核大小
        shape:权重的shape
    @return
        ws:各层权重
        bs:各层偏置
    '''

    def getWeights(self, k, shape):
        ws = []
        bs = []
        for i in range(len(shape) - 1):
            w = tf.truncated_normal([k, k, shape[i], shape[i + 1]], dtype=tf.float32)
            w = tf.Variable(w)
            ws.append(w)
            b = tf.constant(0.1, tf.float32, [shape[i + 1]])
            b = tf.Variable(b)
            bs.append(b)
        return ws, bs

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

    def getHolder(self, obj, shape, name=None):
        if name:
            obj = tf.placeholder(tf.float32, shape)
        else:
            obj = tf.placeholder(tf.float32, shape, name)
        return obj

    '''
    @about
        激活函数
        指定数据和激活函数类型，对数据进行激活
    @param
        input:输入数据
        type:激活函数
    @return
        激活后的数据
    '''

    def active(self, input, type):
        lst = ['relu', 'leaky_relu', 'softmax', 'sigmoid', 'tanh']
        assert (type in lst)
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
            assert (False)

    '''
    @about
        损失函数，根据指定误差计算方式计算误差
    @param
        input:预测值
        output:目标值
        type:损失函数类型
    @return
        损失值
    '''

    def loss(self, input, output, type='l1'):
        self.__logger.debug('computing loss...')
        lst = ['l1', 'cross_entropy']
        assert (type in lst)
        assert (isinstance(input, Tensor))
        assert (input.shape == output.shape)
        if type == 'l1':
            result = tf.reduce_mean(tf.abs(tf.subtract(input, output)))
        elif type == 'cross_entropy':
            # 此方式表现不佳
            result = tf.reduce_mean(-tf.reduce_sum(output * tf.log(input)))
        else:
            assert (False)
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

    def optimizer(self, loss, learning_rate, type):
        self.__logger.debug('building optimizer...')
        if type == 'Adam':
            result = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        elif type == 'Gradient':
            result = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        else:
            assert (False)
        self.__logger.debug('optimizer built.')
        return result
