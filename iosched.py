#!/bin/local/python3
# encoding:utf-8
'''
IO调度模块，用于cache管理

说明：
    此模块的主要任务是开启文件加载线程
    将文件切分为batch粒度，放入训练数据消息队列
    训练时，feed模块从队列中获取，使cache透明
    将所有场景文件组织为一个环形链
    长时间调用此模块后，可保证所有场景有近似的访问频率

基本策略：
两个线程循环执行：
    生产者:加载一个场景到cache中，
    使用cache的主要意义在于可以控制内存占用，垃圾回收不依赖Python的机制
    生产者:按照切片规则将场景切片，取一定数量放入缓冲区（缓冲区满时阻塞），加载下一个场景
    消费者:将缓冲区中的切片放入消息队列，队列满时阻塞

'''

import os
import utils
import queue
import numpy as np

from threading import Thread


# 调度类
class IOsched:
    '''
    @about
        构造函数
    @param
        用于存储场景切片的消息队列
    @return
        None
    '''

    def __init__(self, prop, batchQueue):
        # sample this size from a scene
        self.__sampleNum = 25
        self.__logger = utils.getLogger()
        # buf size
        self.__maxBufSize = 100
        self.__buf = queue.Queue(self.__maxBufSize)
        # send batchs to this queue
        self.__data_queue = batchQueue
        self.__mode = prop.queryAttr('mode')
        assert (self.__mode == 'infer' or self.__mode == 'train')
        # cnn name
        self.__cnn_name = prop.queryAttr('cnn_name')
        self.__batch_n = prop.queryAttr('batch_n')
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        self.__features = prop.queryAttr('features')
        self.__ifeatures = prop.queryAttr('ifeatures')
        self.__batch_c = self.__features + self.__ifeatures
        self.__ground_truth = prop.queryAttr('ground_truth')
        # cache
        self.__cache = prop.queryAttr('cache')
        # size of image
        self.__ih = self.__iw = 0
        self.__input_dir = prop.queryAttr('data_dir')
        self.__scene_name = os.listdir(self.__input_dir)
        if self.__mode == 'train':
            self.__directories = [(self.__input_dir + itm + '/') for itm in self.__scene_name]
        else:
            assert (self.__mode == 'infer')
            self.__directories = [self.__input_dir]

    '''
    @about
        调度开启入口
    @param
        None
    @return
        None
    '''

    def start(self):
        # 创建两线程
        producer = Thread(target=self.produce, args=(), name='Producer')
        consumer = Thread(target=self.consume, args=(), name='Consumer')

        # 守护线程
        producer.setDaemon(True)
        consumer.setDaemon(True)

        # 开启
        producer.start()
        consumer.start()

    '''
    @about
        生产者线程主函数
        按照顺序向cache中调入场景，将场景切片，放入缓冲区中
        阻塞模式
    @param
        None
    @return
        None
    '''

    def produce(self):
        # md5 not match
        self.__abondon = []
        # scenes visited
        self.__visited = []
        # scene ptr
        ptr = 0
        while True:
            image = {}
            txt = {}
            path = self.__directories[ptr]
            # get a scene from cache
            # if not visited,check md5
            for i in range(2):
                imgname = self.__cnn_name[i] + '.png'
                txtname = self.__cnn_name[i] + '.txt'
                image[i] = self.getCacheItem(path + imgname)
                txt[i] = self.getCacheItem(path + txtname)
                assert (len(image[i].shape) == 4)
                assert (len(txt[i].shape) == 4)
            assert (txt[0].shape == txt[1].shape)
            assert (image[0].shape == image[1].shape)
            if self.__mode == 'train':
                truthname = self.__ground_truth
                t = self.getCacheItem(path + truthname)
            else:
                assert (self.__mode == 'infer')
                xn = image[0].shape[0]
                xh = image[0].shape[1]
                xw = image[0].shape[2]
                xc = image[0].shape[3]
                t = np.zeros([xn, xh, xw, xc])
            assert (image[0].shape == t.shape)
            scene = {0: image[0], 1: txt[0],
                     2: image[1], 3: txt[1], 4: t}

            del txt
            del image

            for i in range(self.__sampleNum):
                # how to cut this scene into pieces
                layer, offset, size = self.getSplitParams(scene)
                # get a batch from current scene
                batch = self.splitScene(scene, layer, offset, size)
                assert (isinstance(batch, dict))
                self.__buf.put(batch)
            ptr += 1
            if (ptr == len(self.__directories)):
                ptr = 0

    '''
    @about
        消费者线程主函数
        从缓冲区中取出对象，放入消息队列中
        阻塞模式
    @param
        None
    @return
        None
    '''

    def consume(self):
        while True:
            msg = self.__buf.get()
            self.__data_queue.put(msg)

    '''
    @about
        Sigmoid函数
    @param
        x:输入值
    @return
        处理后的结果
    '''

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    '''
    @about
        对场景文件进行切片
    @param
        scene:dict,场景文件
        layer:用于txt的层数
        offset:list起始点[oy,ox]
        size:list大小[sy,sx]
    @return
        value:dict['img_0':img,'txt_0':txt,'img_1':img,'txt_1':txt,'truth':truth]
    '''

    def splitScene(self, scene, layer, offset, size):
        # self.__logger.debug('spliting scene...')
        ret = {}
        bn = self.__batch_n
        bh = self.__batch_h
        bw = self.__batch_w
        bc = self.__batch_c
        features = self.__features
        ifeatures = self.__features
        ret[1] = ret[3] = np.zeros([1, bh, bw, bc])
        # make sure ih and iw's value
        assert (self.__ih == scene[0].shape[1])
        assert (self.__iw == scene[0].shape[2])
        scene[1] = np.reshape(scene[1], [-1, self.__ih, self.__iw, self.__features])
        scene[3] = np.reshape(scene[3], [-1, self.__ih, self.__iw, self.__features])
        sh = offset[0]
        sw = offset[1]
        cols = 3
        ret[0] = utils.slice(scene[0], [0, sh, sw, 0], [1, bh, bw, cols])  # img-0
        ret[1][:, :, :, :features] = utils.slice(scene[1], [layer, sh, sw, 0], [1, bh, bw, self.__features])  # txt-0
        ret[1][:, :, :, features + 0:features + 1] = utils.getLuminance(ret[0])
        ret[1][:, :, :, features + 1:features + 2] = utils.getMagnitude(ret[0])
        ret[2] = utils.slice(scene[2], [0, sh, sw, 0], [1, bh, bw, cols])  # img-1
        ret[3][:, :, :, :features] = utils.slice(scene[3], [layer, sh, sw, 0], [1, bh, bw, self.__features])  # txt-1
        ret[3][:, :, :, features + 0:features + 1] = utils.getLuminance(ret[2])
        ret[3][:, :, :, features + 1:features + 2] = utils.getMagnitude(ret[2])
        ret[4] = utils.slice(scene[4], [0, sh, sw, 0], [1, bh, bw, cols])  # truth

        # preprocess data 
        ret[1] = self.Sigmoid(ret[1])
        ret[3] = self.Sigmoid(ret[3])

        return ret

    '''
    @about
        根据当前模式，返回切片起始位置和大小
    @param
        scene:dict，场景文件
    @return
        layer,offset,size:[dy,dx]
    '''

    def getSplitParams(self, scene):
        imgkey = len(scene.keys()) - 1  # truth img
        ih = scene[imgkey].shape[1]
        iw = scene[imgkey].shape[2]
        bh = self.__batch_h
        bw = self.__batch_w
        bn = self.__batch_n
        # layer of txt
        txtkey = len(scene.keys()) - 2  # caustic txt
        tn = scene[txtkey].shape[0]
        layer = np.random.randint(0, tn)
        if self.__mode == 'infer':
            offset = [0, 0]
            size = [ih, iw]
        else:
            assert (self.__mode == 'train')
            sh = np.random.randint(0, ih - bh)
            sw = np.random.randint(0, iw - bw)
            offset = [sh, sw]
            size = [bh, bw]

            offset = [400, 400]
            layer = 0
        return layer, offset, size

    '''
    @about
        负责与cache交互，从cache中加载数据
    @param
        key:cache的键名，为文件路径
    @return
        value:cache的键值
    '''

    def getCacheItem(self, key):
        try:
            value = self.__cache.get(key)
        except:
            if key[-4:] == '.txt':
                value = utils.readTXT(key)
                if self.__ih != 0 and self.__iw != 0:
                    value = np.reshape(value, [-1, self.__ih, self.__iw, self.__features])
            else:
                assert (key[-4:] == '.png')
                value = utils.readIMG(key)
                if self.__ih == 0 or self.__iw == 0:
                    self.__ih = value.shape[1]
                    self.__iw = value.shape[2]
            self.__cache.add(key, value)
        return value
