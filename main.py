#!/usr/local/bin/python3
#encoding:utf-8
'''
创建各类的对象，并调用完成任务

'''

from cnn import CNN
from prop import Prop
from filter import Filter
from filter import FilterFeed
from mainproc import MainProc
from mainproc import MainFeed

from test import Test
from test import TestFeed

from threading import Thread
from multiprocessing import Process

import platform
if platform.system() == 'Windows':
    from irender import iRender
    import winsound as ws

import utils
import numpy as np
import tensorflow as tf

from iosched import IOsched
from tensorflow.python.framework.ops import Tensor

# Test CNN
from test import Test
from test import TestFeed



def test(queue):
    # init
    prop = Prop()
    prop.updateAttr('queue',queue)

    logger = utils.getLogger()
    logger.info(prop.toString())

    cnn = CNN(prop,queue)

    filter = Test(prop)
    feed = TestFeed(prop)

    #cnn.test(feed,filter)
    cnn.train(feed,filter)
    
    logger.info('done.')
    if platform.system() == 'Windows':
        ws.Beep(600,1200)

def train(queue):
    # init
    prop = Prop()
    prop.updateAttr('queue',queue)

    logger = utils.getLogger()
    logger.info(prop.toString())

    cnn = CNN(prop,queue)

    filter = Filter(prop)
    feed = FilterFeed(prop)

    #filter = Test(prop)
    #feed = TestFeed(prop)

    cnn.train(feed,filter)
    
    logger.info('done.')
    if platform.system() == 'Windows':
        ws.Beep(600,1200)


def predict():
    # init
    prop = Prop()
    cnn = CNN(prop)
    
    #'''
    feed = FilterFeed(prop)
    filter = Filter(prop)
    #'''

    '''
    feed = TestFeed(prop)
    filter = Test(prop)
    '''

    logger = utils.getLogger()
    logger.info(prop.toString())

    input,data,truth = feed.getInputdata()

    predict = cnn.process(feed,filter)

    print(truth[0,0,:5,:])
    print('========')
    print(predict[0,0,:5,:])

    utils.saveImage(predict,'data/test/staircase2/predict.png')
    # predict = utils.readIMG('data/test/staircase2/predict.png')
    # print('========')
    # print(predict[0,0,:5,:])

    # compute loss
    loss = tf.reduce_mean(tf.abs(tf.subtract(truth,predict)))
    with tf.Session() as sess:
        loss = sess.run(loss)
        logger.debug('current loss=%f',loss)

    logger.info('done.')
    if platform.system() == 'Windows':
        ws.Beep(600,1200)

        # display
        utils.displayImage(input,predict,truth)



def main(argv=None):
    queue = utils.getQueue()

    prop = Prop()
    logger = utils.getLogger()

    # train cnn if necessary
    if prop.needTrain():
        logger.info('training...')
        # compute in Thread [back]
        pback = Thread(target=train,args=(queue,),name='BackGround')
        # pback = Thread(target=test,args=(queue,),name='backThread')
        #pback.setDaemon(True)
        pback.start()
        #pback.join()
        # render in Thread [main]
        if platform.system() == 'Windows':
            irender = iRender(queue,prop)
            while True:
                irender.process()

    else:
        logger.info('inferring...')
        pback = Thread(target=predict,args=(),name='BackGround')
        #pback.join()
        pback.start()
        



#if __name__ == '__main__':
#    prop = Prop()
#    irender = iRender(prop)
#    irender.process()
#    #tf.app.run(main,argv=None)
if __name__ == '__main__':
    #main()
    logger = utils.getLogger()
    prop = Prop()
    logger.info(prop.toString())
    msg_queue = utils.getQueue('msg')
    data_queue = utils.getQueue('data')

    sched = IOsched(prop,data_queue)
    #producer = Thread(target=sched.produce,args=(),name='Producer')
    #consumer = Thread(target=sched.consume,args=(),name='Consumer')
    #producer.start()
    #consumer.start()
    sched.start()

    mainproc = MainProc(prop,msg_queue)
    mainfeed = MainFeed(prop,data_queue)


    pred = mainproc.preprocess(prop,None)
    result = mainproc.process(mainfeed,pred)
    mainproc.postprocess(result)

    #logger.debug('hello,world')
    
    
