#!/bin/local/python3
# encoding:utf-8
'''
程度的主要流程，合并两条滤波线路并
'''

import utils
import platform
import tensorflow as tf

from proc import Proc
from proc import Feed
from filter import Filter
from shared import Shared
# from eprogress import LineProgress
from filter import FilterFeed
from threading import Thread
from cnn import CNN

'''
负责整个程序流程
创建卷积网络对象，并进行训练或预测
'''


class MainProc(Proc):
    '''
    @about
        构造函数，只接受参数，不进行初始化
        初始化在线程内完成
    @param
        prop:程序配置模块
    @return
        None
    '''

    def __init__(self, prop, feed, msg_queue=None):
        sd = Shared()
        self.__logger = sd.getLogger()
        self.__prop = prop
        self.__feed = feed
        self.__msg_queue = msg_queue

    '''
    @about
        创建两个卷积网络对象
    @param
        prop:程序配置对象
        feed:数据提供对象
        msg_queue:消息队列
    @return
        None
    '''

    def init(self):
        self.__isTrain = self.__prop.needTrain()
        self.__sess = tf.Session()
        self.__prop.updateAttr('session', self.__sess)

        self.__output_path = self.__prop.queryAttr('data_dir')
        self.__model_path = self.__prop.queryAttr('model_path')
        self.__ckpt_name = self.__prop.queryAttr('ckpt_name')
        self.__cnn_name = self.__prop.queryAttr('cnn_name')
        self.__batch_n = self.__prop.queryAttr('batch_n')
        self.__batch_h = self.__prop.queryAttr('batch_h')
        self.__batch_w = self.__prop.queryAttr('batch_w')
        self.__batch_c = self.__prop.queryAttr('sfeatures') + self.__prop.queryAttr('ifeatures')
        self.__cols = self.__prop.queryAttr('cols')

        # global photon cnn
        self.__global_fea = tf.placeholder(tf.float32, [self.__batch_n, self.__batch_h, self.__batch_w, self.__batch_c])
        self.__global_img = tf.placeholder(tf.float32, [self.__batch_n, self.__batch_h, self.__batch_w, self.__cols])
        self.__globalCNN = CNN(self.__prop, self.__cnn_name[0])
        self.__globalCNN.build(self.__global_img, self.__global_fea)
        # self.__globalCNN.init()

        # caustic photon cnn
        self.__caustic_fea = tf.placeholder(tf.float32,
                                            [self.__batch_n, self.__batch_h, self.__batch_w, self.__batch_c])
        self.__caustic_img = tf.placeholder(tf.float32, [self.__batch_n, self.__batch_h, self.__batch_w, self.__cols])
        self.__causticCNN = CNN(self.__prop, self.__cnn_name[1])
        self.__causticCNN.build(self.__caustic_img, self.__caustic_fea)
        # self.__causticCNN.init()

        # other configuration in train mode
        if self.__isTrain:
            self.__loss_func = self.__prop.queryAttr('loss_func')
            self.__max_round = self.__prop.queryAttr('max_round')
            self.__save_round = self.__prop.queryAttr('save_round')
            self.__learning_rate = self.__prop.queryAttr('learning_rate')

    '''
    @about
        线程调用入口
        封装此模块主要流程
    @param
        None
    @return
        None
    '''

    def run(self):
        self.init()
        predict = self.preprocess(self.__prop)
        result = self.process(self.__feed, predict)
        self.postprocess(result)

    '''
    @about
        开启守护线程，执行此模块主要逻辑
    @param
        None
    @return
        None
    '''

    def start(self):
        wrk = Thread(target=self.run, args=(), name='Worker')
        wrk.setDaemon(True)
        wrk.start()

    '''
    @about
        向消息队列中发送数据
    @param
        msg:[current status %,current loss]
    @return
        None
    '''

    def sendMsg(self, msg):
        if platform.system() != 'Windows':
            return
        try:
            self.__msg_queue.put(msg)
        except:
            self.__logger.error('message queue is not specified.')

    '''
        @about
            优化器
        @param
            loss:损失值
            learning_rate:学习率
            type:优化器类型
        @return
            优化器
    '''

    def __getOptimizer(self, loss, learning_rate, type):
        self.__logger.debug('building optimizer...')
        if type == 'Adam':
            result = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        elif type == 'Gradient':
            result = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        else:
            raise NotImplementedError
        self.__logger.debug('optimizer built.')
        return result

    '''
    @about
        两个卷积网络进行滤波，然后合并结果
    @param
        prop:
        input2:
    @return
        预测值，Tensor
    '''

    def preprocess(self, prop):
        self.__logger.debug('preprocessing...')
        # filter images separately
        glob = self.__globalCNN.process(Filter(prop))
        caus = self.__causticCNN.process(Filter(prop))

        assert (glob.shape == caus.shape)
        # predict should be clipped to [0,255]
        predict = glob + caus
        # predict = caus

        '''
        pmax = tf.reduce_max(predict,-1)
        n = psum.shape[0]
        h = psum.shape[1]
        w = psum.shape[2]
        pmax = tf.reshape(pmax,[n,h,w,1])
        
        # scale each channel to [0,255]
        predict = predict / pmax * 255
        '''
        # self.__predict = predict
        return predict

    '''
    @about
        主要处理流程，处理来自preprocess的predict
    @param
        feed:   数据填充模块
        predict:预测值
    @return
        处理过的predict
    '''

    def process(self, feed, predict):
        self.__logger.debug('processing...')
        ishape, tshape = feed.getBatchShapes()

        # model storage path
        ckpt_global = self.__model_path + self.__cnn_name[0] + '/'
        ckpt_caustic = self.__model_path + self.__cnn_name[1] + '/'

        # train when in train mode
        if self.__isTrain:
            self.__logger.debug('training...')

            truth = tf.placeholder(tf.float32, ishape)
            loss = tf.reduce_mean(tf.abs(tf.subtract(predict, truth)))
            # step = tf.train.AdamOptimizer(self.__learning_rate).minimize(loss)
            step = self.__getOptimizer(loss, self.__learning_rate, self.__loss_func)

            # 恢复与随机初始化两网络
            self.__globalCNN.init(ckpt_global)
            self.__causticCNN.init(ckpt_caustic)

            # 用于绘制进度条
            # bar = LineProgress(title='status', total=self.__max_round)

            # 训练
            for i in range(self.__max_round):

                gi, gf, ci, cf, gt = feed.next_batch()
                self.__sess.run(step, feed_dict={self.__caustic_img: ci, self.__global_img: gi,
                                                 self.__caustic_fea: cf, self.__global_fea: gf,
                                                 truth: gt})
                xloss = self.__sess.run(loss, feed_dict={self.__caustic_img: ci, self.__global_img: gi,
                                                         self.__caustic_fea: cf, self.__global_fea: gf,
                                                         truth: gt})

                self.sendMsg([i / self.__max_round, xloss])

                # xpred = self.__sess.run(predict, feed_dict={self.__caustic_img: ci, self.__global_img: gi,
                #                                            self.__caustic_fea: cf, self.__global_fea: gf,
                #                                            truth: gt})
                # utils.displayImage(xpred)

                print('round:%d of %d,loss: %f...' % (i + 1, self.__max_round, xloss))
                self.__logger.info('round:%d of %d,loss:%f...' % (i + 1, self.__max_round, xloss))
                # print("status: {:.2f}%".format(float((i + 1) / self.__max_round)), end="\r")

                # bar.update((i + 1) / self.__max_round * 100)

                # 保存结果
                if i % self.__save_round == (self.__save_round - 1):
                    self.__globalCNN.save(ckpt_global, self.__ckpt_name, self.__save_round)
                    self.__causticCNN.save(ckpt_caustic, self.__ckpt_name, self.__save_round)


        # infer模式下直接输出
        else:
            self.__logger.debug('inferring...')

            # 恢复与随机初始化两网络
            self.__globalCNN.init(ckpt_global)
            self.__causticCNN.init(ckpt_caustic)

            gi, gf, ci, cf = feed.getInputdata()
            result = self.__sess.run(predict, feed_dict={self.__caustic_img: ci, self.__global_img: gi,
                                                         self.__caustic_fea: cf, self.__global_fea: gf})
            return result

        return None

    '''
    @about
        process之后执行
    @param
        input:process的输出
    @return
        None
    '''

    def postprocess(self, input1):
        sd = Shared()
        self.__logger.debug('postprocessing...')
        if not self.__isTrain:
            # path of test data file
            save_path = self.__output_path
            utils.saveImage(input1, save_path + 'infer.png')
            # utils.displayImage(input)

        sd.setFlag('nowExit', True)
        print('done')


'''
配合mainproc对象使用，为其提供数据
'''


class MainFeed(Feed):
    '''
    @about
        构造函数
    @param
        prop：程序配置对象
    @return
        None
    '''

    def __init__(self, prop, data_queue):
        # shape of input data
        self.__cols = prop.queryAttr('cols')
        self.__batch_n = prop.queryAttr('batch_n')
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        self.__sfeatures = prop.queryAttr('sfeatures')
        self.__ifeatures = prop.queryAttr('ifeatures')
        self.__batch_c = self.__sfeatures + self.__ifeatures

        sd = Shared()
        self.__logger = sd.getLogger()
        # two networks
        cnns = prop.queryAttr('cnn_name')

        self.__data_queue = data_queue

        # network-1
        self.__feed1 = FilterFeed(prop)
        img_name = cnns[0] + '.png'
        txt_name = cnns[0] + '.txt'
        conf_name = cnns[0] + '.json'
        self.__feed1.setInputName(img_name, txt_name, conf_name)

        # network-2
        self.__feed2 = FilterFeed(prop)
        img_name = cnns[1] + '.png'
        txt_name = cnns[1] + '.txt'
        conf_name = cnns[1] + '.json'
        self.__feed2.setInputName(img_name, txt_name, conf_name)

    '''
    @about
        获取输入数据形状
    @param
        None
    @return
        shape1:image的形状
        shape2:text的形状
    '''

    def getBatchShapes(self):
        cols = 3
        bn = self.__batch_n
        bh = self.__batch_h
        bw = self.__batch_w
        bc = self.__batch_c
        return [bn, bh, bw, cols], [bn, bh, bw, bc]

    '''
    @about
        计算md5,仅测试用
    @param
         string:输入字符串
    @return
         md5
    '''

    def __test_md5(self, string):
        import hashlib
        vmd5 = hashlib.md5()
        vmd5.update(string.encode('utf8'))
        ret = vmd5.hexdigest()
        return ret

    '''
    @about
        用于训练时输入数据
    @param
        None
    @return
        gi:global images
        gf:global features
        ci:caustic images
        cf:caustic features
        gt:ground truth
    '''

    def next_batch(self):
        ret = self.__data_queue.get()
        assert (len(ret) == 5)
        gi = ret[0]
        gf = ret[1]
        ci = ret[2]
        cf = ret[3]
        truth = ret[4]
        return gi, gf, ci, cf, truth

    '''
    @about
        文件校验
    @param
        
    @return
       True/False 
    '''

    def checkMatch(self, a, b, c):
        return super().checkMatch(a, b, c)

    '''
    @about
        用于预测模式数据输入
    @param
        None
    @return
        gi,ci:全局光子和焦散光子图 的输入图像
        gf,cf:全局光子和焦散光子图 的对应特征
    '''

    def getInputdata(self):
        ret = self.__data_queue.get()
        gi = ret[0]
        gf = ret[1]
        ci = ret[2]
        cf = ret[3]
        return gi, gf, ci, cf
