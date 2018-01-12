# encoding:utf-8
'''
Created on 2017-12-6

@author: deyuz
'''

import os
import queue
from threading import Thread

import model
import tensorflow as tf
import utils
import winsound as ws


# 初始化参数
def initParams():
    params = {}
    params['log'] = utils.getLogger('stdout')
    # 从配置文件中读取参数
    params['mode'] = utils.getProperties(name='mode',default='train')
    params['train_data_dir'] = utils.getProperties(name='train_data_dir',default='./')
    params['test_data_dir'] = utils.getProperties(name='test_data_dir',default='./')
    params['model_dir'] = utils.getProperties(name='model_dir',default='./')
    params['queue'] = queue.Queue(utils.getProperties(name='queueSize',default=1))
    params['k'] = utils.getProperties(name='k',default=11)
    params['batch_width'] = utils.getProperties(name='batch_width',default=65)
    params['batch_height'] = utils.getProperties(name='batch_height',default=65)
    params['maxCacheSize'] = utils.getProperties(name='maxCacheSize',default=1)
    params['learning_rate'] = utils.getProperties(name='learning_rate',default=1e-4)
    params['max_round'] = utils.getProperties(name='max_round',default=10)
    # weights_shape应比conv_shape多一个元素
    params['weights_shape'] = utils.getProperties(name='weights',group_name='shape',default=[8,100,100])
    params['weights_shape'].append(params['k'] ** 2)
    params['conv_shape'] = utils.getProperties(name='conv',group_name='shape',default=[5, 5, 5])
    return params
    

# 主函数
def main(argv=None):
    # 初始化参数
    params = initParams()
    
    # 生成渲染线程
    prender = Thread(target=utils.onRender, args=(params['queue'],), name='render')
    # 生成后台线程
    if params['mode'] == 'train':
        pmodel = Thread(target=model.onTrain, args=(params,), name='train')
        prender.setDaemon(True)
    else:
        assert(params['mode'] == 'predict')
        pmodel = Thread(target=model.onPredict, args=(params,), name='predict')
    
    # 开启线程
    pmodel.start()
    prender.start()
    # 等待线程退出
    pmodel.join()
    # 完成声音提示
    ws.Beep(600, 1200)
    return



if __name__ == '__main__':
    tf.app.run(main=main, argv=())
    #try:
    #     tf.app.run(main=main, argv=())
    #except:
    #     print('ignore an exception')
    print('done')
    #os.system('shutdown -s -t 60')




