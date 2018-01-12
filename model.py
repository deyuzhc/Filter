# encoding:utf-8
'''
Created on 2017-12-6

@author: deyuz
'''

import os
import time
import math

import numpy as np
import tensorflow as tf
import utils


# 全局参数
params = {}

# 程序初始化需要的参数
def initParam(inputs):
    global params
    len1 = len(inputs)
    params = inputs
    utils.initParams(inputs)
    params['batchs'] = utils.getProperties(name='batchs',default=2)
    # 输入文本数据的shape default:[?,65,65,8]
    params['data_shape'] = [-1, inputs['batch_height'], inputs['batch_width'], inputs['batch_channels']]
    # 处理输入数据维度指定前后不一致的情况
    if inputs['batch_channels'] != params['weights_shape'][0]:
        params['log'].warn('channels of the model input layer mismatch the value read from scene files,update the previous one.')
        params['weights_shape'][0] = params['batch_channels']
    params['X1'], params['X2'], params['Y'] = initXY(params['batch_channels'])
    params['Ws'], params['bs'] = initWeights()
    params['sess'] = tf.Session()
    params['sess'].run(tf.global_variables_initializer())
    len2 = len(inputs)
    params['log'].debug('add arguments in model:%d' % (len2 - len1))
    

# 2D conv
def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, padding='SAME', strides=[1, 1, 1, 1], name=name)


# 初始化 X(shape = 2D) & Y
def initXY(txt_channels=8,img_channels=3):
    global X1, X2, Y
    X1 = tf.placeholder(dtype=tf.float32, shape=[None,txt_channels], name='txt_input')
    X2 = tf.placeholder(dtype=tf.float32, shape=[None, img_channels], name="image_input")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, img_channels], name='image_truth')
    return X1, X2, Y


# 初始化 W & b
# weights_shape=[8,100,100,100,3]
# conv_shape =[ 5, 5, 5, 5]
def initWeights():
    Ws = []
    bs = []
    for i in range(len(params['weights_shape']) - 1):
        with tf.variable_scope('conv_layer-' + str(i)):
            W = tf.get_variable(name='W', shape=[params['conv_shape'][i], params['conv_shape'][i],
                params['weights_shape'][i], params['weights_shape'][i + 1]],
                dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, dtype=tf.float32))
            b = tf.get_variable(name='b', shape=[params['weights_shape'][i + 1]], dtype=tf.float32, initializer=
                tf.truncated_normal_initializer(mean=0.0, dtype=tf.float32))
        Ws.append(W)
        bs.append(b)
    return Ws, bs


# 输入X,Y,Ws,bs
# 输出[?,h,w,kxk] softmax之后的所有像素的滤波核
def outputfilters():
    params['log'].info('generating filters...')
    # softmax output
    h = params['X1']
    # reshape h
    h = tf.reshape(h,params['data_shape'])
    for i in range(len(params['bs'])):
        h = tf.nn.relu(conv2d(h, params['Ws'][i], 'conv2d-' + str(i)) + params['bs'][i], name='relu-' + str(i))
    h = tf.nn.softmax(h, name='filters')
    params['log'].info('filters generated.')
    return h
    
    
# 滤波函数:若有更高效的算法此函数应被替换
# src [?,85,85,3] dtype=float32
# kernels [?,65,65,kxk]
# 输出 [?,h,w,3]
def filterImage(src, kernels):
    params['log'].info('filtering images...')
    outcolor = []
    for n in range(params['batchs']):
        params['log'].debug('filtering batch:%d,%d left...' % (n + 1,params['batchs'] - n - 1))
        for y in range(params['batch_height']):
            for x in range(params['batch_width']):
                kernel = tf.slice(kernels, [n, y, x, 0], [1, 1, 1, params['k'] ** 2])  # [1,1,1,kxk]
                kernel = tf.reshape(kernel, [-1, 1])  # [kxk,1]
                colors = tf.slice(src, [n, y, x , 0], [1, params['k'], params['k'], 3])  # [1,k,k,3]
                colors = tf.reshape(colors, [-1, 3])  # [kxk,3]
                outcolor.append(tf.reduce_sum(colors * kernel, axis=0))
    outcolor = tf.reshape(outcolor, [params['batchs'], params['batch_height'], params['batch_width'], 3])
    params['log'].info('images filtered.')
    return outcolor


# 输入图片，根据outputfilter输出的滤波器拟合颜色值
def predict():
    params['log'].info('generating output images...')
    # 获取所有像素的滤波核
    filters = outputfilters()  # shape=[?,h,w,kxk]，已对最后一维进行归一化
    # 填充原图
    X_mat = tf.reshape(params['X2'], [-1, params['batch_height'], params['batch_width'], 3])
    kr = int(params['k'] / 2)  # 滤波核半径
    X_mat = tf.cast(X_mat, tf.float32)
    image_padded = tf.pad(X_mat, [[0, 0], [kr, kr], [kr, kr], [0, 0]], 'SYMMETRIC')
    # 返回滤波后的值 [?,h,w,3]
    #X_mat=tf.reshape(params['X2'],[-1,h+k-1,w+k-1,3])
    image_filtered = filterImage(image_padded, filters)
    params['log'].info('output images generated.') 
    return image_filtered


# 误差函数
# image_pred :拟合颜色值 y [?,h,w,3]
# image_truth:目标颜色值 Y [?,h,w,3]
def loss(image_pred):
    params['log'].info('computing loss...')
    y = tf.reshape(image_pred, [-1, 3])
    entropy = tf.reduce_mean(tf.abs(tf.subtract(y, params['Y'])))
    params['log'].info('loss computed.')
    return entropy


# 评价准确率
def accuracy(image_pred, image_truth):
    y = tf.reshape(image_pred, [-1, 3])
    Y = image_truth  # [-1,3]
    ysub = tf.subtract(y, Y)
    diffs = tf.reduce_sum(tf.multiply(ysub, ysub), axis=1)
    zeros = tf.constant(0, dtype=tf.int32, shape=[params['batch_height'] * params['batch_width'], 1])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(diffs, tf.int32), zeros), tf.float32))
    return accuracy


# 优化函数
def optimizer(entropy):
    params['log'].info('building optimizer...')
    step = tf.train.AdamOptimizer(params['learning_rate']).minimize(entropy)
    params['sess'].run(tf.global_variables_initializer())
    params['log'].info('optimizer built.')
    return step
    

# 保存模型
def save(path):
    try:
        os.makedirs(path)
        params['log'].info('directory \'%s\' was made for model saving.' % path)
    except OSError:
        if not os.path.isdir(path):
            params['log'].warn('target path is not a directory,skip saving.')
            return
    params['log'].info('saving model to \'%s\'...' % path)
    saver = tf.train.Saver()
    saver.save(params['sess'], path + 'model')
    params['log'].info('model saved.')


# 恢复模型
def restore(path_dir):
    if os.path.exists(path_dir + 'checkpoint'):
        ckpt = tf.train.latest_checkpoint(path_dir)
        params['log'].debug("checkpoint path is '%s'" % ckpt)
        saver = tf.train.Saver(save_relative_paths=True)
        saver.restore(params['sess'], ckpt)
        params['log'].info('model restored from the latest checkpoint.')
    else:
        params['log'].info('checkpoint not found,initializing variables...')

# 训练函数
def onTrain(inputs):
    # 初始化参数
    initParam(inputs)
    params['log'].debug('onTrain')  
    # 开始计时
    st = time.clock()  
    # 建立模型
    image_pred = predict()
    entropy = loss(image_pred)
    step = optimizer(entropy)   
    # 恢复数据
    restore(params['model_dir'])   
    # 训练
    params['log'].info('start training...')
    global dt
    dt = 9.9
    for i in range(params['max_round']):
        params['log'].info('current round: %d,%d left...' % (i + 1, params['max_round'] - i - 1))
        bef = time.clock()
        x1, x2, y = utils.next_batch(params['batchs'])
        params['sess'].run(step, feed_dict={X1:x1, X2:x2, Y:y})
        if i % 50 == 0:
            save(params['model_dir'])
        if i % 1 == 0:
            params['log'].debug('predicting...')
            #image_res = params['sess'].run(image_pred, feed_dict={X1:x1,
            #X2:x2})
            # image_res = np.reshape(image_res, (params['height'],
            # params['width'], 3))
            # params['queue'].put(image_snd)
            #params['log'].info('accuracy = %.8f,time left: %1.1fm.' %
            #(params['sess'].run(accuracy(image_res, y)), dt *
            #(params['max_round'] - 1) / 60))
            params['log'].info('loss = %.8f,time left: %1.1fm.' % (params['sess'].run(entropy,feed_dict={X1:x1,X2:x2,Y:y}), dt * (params['max_round'] - 1 - i) / 60))
        aft = time.clock()
        dt = float(aft - bef)        
    # 保存结果
    save(params['model_dir'])    
    # 终止计时
    ed = time.clock()
    params['log'].info('total time cost: %f min.' % ((ed - st) / 60.0))


# 预测函数
def onPredict(inputs):
    # 初始化参数
    initParam(inputs)
    params['log'].debug('onPredict')
    # 开始计时
    st = time.clock()
    # 恢复数据
    restore(params['model_dir'])
    # h,w图像大小
    # tbatchs:X1;ibatchs:X2
    # 随机选择一张图，进行整张输入与预测
    h,w,tbatchs,ibatchs = utils.splitScene(params['batch_height'],params['batch_width'])
    # 若splitScene指定batch大小，在此断言
    #print(tbatchs[0].shape,ibatchs[0].shape)
    # 滤波后的图像
    finalImage = np.zeros((h,w,3))
    # batch大小
    assert(len(tbatchs) > 0)
    bh = ibatchs[0].shape[1]    #params['batch_height']
    bw = ibatchs[0].shape[2]    #params['batch_width']
    bc = tbatchs[0].shape[3]
    # 建立模型
    params['log'].debug('update [data_shape]')
    params['data_shape'][1] = bh
    params['data_shape'][2] = bw
    params['data_shape'][3] = bc
    image_pred = predict()
    # 碎片数
    nh = math.ceil(float(h) / bh)
    nw = math.ceil(float(w) / bw)
    params['log'].debug('nh=%d;nw=%d' % (nh,nw))
    params['log'].info('generating and reconstructing output images,please wait...')
    for y in range(nh):
        sy = min(y * bh,h - bh)
        for x in range(nw):
            sx = min(x * bw,w - bw)
            finalImage[sy:sy + bh,sx:sx + bw,:] = params['sess'].run(image_pred,
            feed_dict={X1:np.reshape(tbatchs[y * nw + x],[-1,bc]),X2:np.reshape(ibatchs[y * nw + x],[-1,3])})#ibatchs[y * nw + x]
            
            image_snd = finalImage[sy:sy + bh,sx:sx + bw,:]
            image_snd = np.reshape(image_snd,[bh,bw,3])
            params['queue'].put(image_snd)
    params['log'].info('output images generated.')
    # 向渲染线程发送内容
    #image_snd = np.reshape(finalImage, (h, w, 3))
    #params['queue'].put(image_snd)
    ed = time.clock()
    params['log'].info('total time cost: %d s.' % int(ed - st))
