#encoding:utf-8
'''
程度的主要流程，合并两条滤波线路并
'''

from proc import Proc
from proc import Feed
from filter import Filter
from filter import FilterFeed
from cnn import CNN



import utils
import platform
import tensorflow as tf
import numpy as np

class MainProc(Proc):
    '''
    @about
        创建两个卷积网络对象
    @param
        prop:程序配置模块
    @return
        None
    '''
    def __init__(self,prop,msg_queue=None):
        self.__logger = utils.getLogger()
        self.__isTrain = prop.needTrain()
        self.__sess = prop.queryAttr('session')
        
        self.__msg_queue = msg_queue

        self.__output_path = prop.queryAttr('data_dir')
        self.__model_path = prop.queryAttr('model_path')
        self.__model_name = prop.queryAttr('model_name')
        self.__cnn_name = prop.queryAttr('cnn_name')
        self.__batch_n = prop.queryAttr('batch_n')
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        self.__batch_c = prop.queryAttr('features') + prop.queryAttr('ifeatures')
        self.__cols = prop.queryAttr('cols')

        # global photon cnn
        self.__global_fea = tf.placeholder(tf.float32,[self.__batch_n,self.__batch_h,self.__batch_w,self.__batch_c])
        self.__global_img = tf.placeholder(tf.float32,[self.__batch_n,self.__batch_h,self.__batch_w,self.__cols])
        self.__globalCNN = CNN(prop,self.__cnn_name[0])
        self.__globalCNN.getInput(self.__global_img,self.__global_fea)
        #self.__globalCNN.init()


        # caustic photon cnn
        self.__caustic_fea = tf.placeholder(tf.float32,[self.__batch_n,self.__batch_h,self.__batch_w,self.__batch_c])
        self.__caustic_img = tf.placeholder(tf.float32,[self.__batch_n,self.__batch_h,self.__batch_w,self.__cols])
        self.__causticCNN = CNN(prop,self.__cnn_name[1])
        self.__causticCNN.getInput(self.__caustic_img,self.__caustic_fea)
        #self.__causticCNN.init()

        # other configuration in train mode
        if self.__isTrain:
            self.__max_round = prop.queryAttr('max_round')
            self.__learning_rate = prop.queryAttr('learning_rate')


    '''
    @about
        向消息队列中发送数据
    @param
        msg:[current status %,current loss]
    @return
        None
    '''
    def sendMsg(self,msg):
        if platform.system() != 'Windows':
            return
        try:
            self.__msg_queue.put(msg)
        except:
            self.__logger.error('message queue is not specified.')


    '''
    @about
        两个卷积网络进行滤波，然后合并结果
    @param
        prop:
        input2:
    @return
        预测值，Tensor
    '''
    def preprocess(self, prop, input2=None):
        self.__logger.debug('preprocessing...')
        # filter images separately
        glob = self.__globalCNN.process(Filter(prop))
        caus = self.__causticCNN.process(Filter(prop))

        assert(glob.shape == caus.shape)
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
    def process(self,feed,predict):
        self.__logger.debug('processing...')
        ishape,tshape = feed.getBatchShapes()

        # model storage path
        ckpt_global = self.__model_path + self.__cnn_name[0] + '/'
        ckpt_caustic = self.__model_path + self.__cnn_name[1] + '/'

        # train when in train mode
        if self.__isTrain:
            self.__logger.debug('training...')
            
            truth = tf.placeholder(tf.float32,ishape)
            loss = tf.reduce_mean(tf.abs(tf.subtract(predict,truth)))
            step = tf.train.AdamOptimizer(self.__learning_rate).minimize(loss)

            # init these two network randomly
            self.__sess.run(tf.global_variables_initializer())
            self.__globalCNN.init(ckpt_global)
            self.__causticCNN.init(ckpt_caustic)

            #self.__globalCNN.restore(path_global)
            #self.__causticCNN.restore(path_caustic)

            # restore networks
            for i in range(self.__max_round):

                gi,gf,ci,cf,gt = feed.next_batch()
                self.__sess.run(step,feed_dict={self.__caustic_img:ci,self.__global_img:gi,
                                                self.__caustic_fea:cf,self.__global_fea:gf,
                                                truth:gt})
                xloss = self.__sess.run(loss,feed_dict={self.__caustic_img:ci,self.__global_img:gi,
                                                        self.__caustic_fea:cf,self.__global_fea:gf,
                                                        truth:gt})

                self.sendMsg([i / max(0,self.__max_round),xloss])
                
                xpred = self.__sess.run(predict,feed_dict={self.__caustic_img:ci,self.__global_img:gi,
                                                        self.__caustic_fea:cf,self.__global_fea:gf,
                                                        truth:gt})

                # print(xpred.shape)
                #xpred = np.reshape(xpred,[100,100,3])
                #utils.saveImage(xpred,'tmp/predict-'+ str(i) +'.png')

                if self.__max_round > 0:
                    self.__logger.info('round:%d of %d,loss:%f...' % (i + 1,self.__max_round,xloss))
                else:
                    self.__logger.info('round:%d of inf,loss:%f...' % (i + 1,xloss))

                # save result
                if i % 100 == 99:
                    self.__globalCNN.save(ckpt_global,self.__model_name)
                    self.__causticCNN.save(ckpt_caustic,self.__model_name)

        # output directly when in infer mode
        else:
            self.__logger.debug('infering...')
            # init these two network randomly
            self.__sess.run(tf.global_variables_initializer())

            self.__globalCNN.init(ckpt_global)
            self.__causticCNN.init(ckpt_caustic)

            #self.__globalCNN.restore(ckpt_global)
            #self.__causticCNN.restore(ckpt_caustic)

            gi,gf,ci,cf = feed.getInputdata()
            result = self.__sess.run(predict,feed_dict={self.__caustic_img:ci,self.__global_img:gi,
                                                        self.__caustic_fea:cf,self.__global_fea:gf})

            return result

    '''
    @about
        process之后执行
    @param
        input:process的输出
    @return
        None
    '''
    def postprocess(self,input):
        self.__logger.debug('postprocessing...')
        # path of test data file
        save_path = self.__output_path
        utils.saveImage(input,save_path + 'infer.png')
        #utils.displayImage(input)




class MainFeed(Feed):

    '''
    @about
        构造函数
    @param
        prop：程序配置对象
    @return
    '''
    def __init__(self,prop,data_queue):
        # shape of input data
        self.__cols = prop.queryAttr('cols')
        self.__batch_n = prop.queryAttr('batch_n')
        self.__batch_h = prop.queryAttr('batch_h')
        self.__batch_w = prop.queryAttr('batch_w')
        self.__features = prop.queryAttr('features')
        self.__ifeatures = prop.queryAttr('ifeatures')
        self.__batch_c = self.__features + self.__ifeatures

        self.__logger = utils.getLogger()
        # two networks
        cnns = prop.queryAttr('cnn_name')

        self.__data_queue = data_queue

        # network-1
        self.__feed1 = FilterFeed(prop)
        img_name = cnns[0] + '.png'
        txt_name = cnns[0] + '.txt'
        conf_name = cnns[0] + '.json'
        self.__feed1.setInputName(img_name,txt_name,conf_name)

        # network-2
        self.__feed2 = FilterFeed(prop)
        img_name = cnns[1] + '.png'
        txt_name = cnns[1] + '.txt'
        conf_name = cnns[1] + '.json'
        self.__feed2.setInputName(img_name,txt_name,conf_name)


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
        return [bn,bh,bw,cols],[bn,bh,bw,bc]

    '''
    @about
        计算md5,仅测试用
    @param
         string:输入字符串
    @return
         md5
    '''
    def test_md5(self,string):
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
        # self.__logger.debug('waiting,queue size:%d...' % self.__data_queue.qsize())
        ret = self.__data_queue.get()
        assert(len(ret) == 5)
        gi = ret[0]
        gf = ret[1]
        ci = ret[2]
        cf = ret[3]
        truth = ret[4]
        #print(self.test_md5(str(gi)))
        #print(self.test_md5(str(gf)))
        #print(self.test_md5(str(ci)))
        #print(self.test_md5(str(cf)))
        #print(self.test_md5(str(truth)))
        
        # @deprecated
        ''' 
        INT_MAX = 2147483647
        seed = np.random.randint(0,INT_MAX)
        ci,cf,_ = self.__feed2.next_batch(seed)
        gi,gf,gt = self.__feed1.next_batch(seed)
        '''
        return gi,gf,ci,cf,truth

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
        #gi,gf,_ = self.__feed1.getInputdata()
        #ci,cf,_ = self.__feed2.getInputdata()
        ret = self.__data_queue.get()
        gi = ret[0]
        gf = ret[1]
        ci = ret[2]
        cf = ret[3]
        return gi,gf,ci,cf


    '''
    @about
    @param
    @return
    '''
    def toString(self):
        return super().toString()