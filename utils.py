# encoding:utf-8
'''
Created on 2017-12-7

@author: deyuz
'''
import logging
import sys
import json
import math

from PIL import Image

import matplotlib.pyplot as plt
import scipy.misc as spm
import numpy as np


# 全局参数
params = {}

# 初始化参数
def initParams(inputs=None):
    global params
    len1 = len(inputs)
    params = inputs
    params['train_config'] = readJson(params['train_data_dir'] + '/config.json')
    params['test_config'] = readJson(params['test_data_dir'] + '/config.json')
    params['sceneCache'] = {}  # 每项为一个场景对应的字典
    params['batch_channels'] = readJson((params['train_data_dir'] + params['train_config']['scenes'][0]['path'] + '/config.json'))['attributes']['channels']
    len2 = len(inputs)
    params['log'].debug('add arguments in utils:%d' % (len2 - len1))
    

# 返回全局logger
def getLogger(mode='stdout'):
    # 首先初始化log，以为全局所用
    logger = logging.getLogger('logging')
    if mode == 'file':
        handler = logging.FileHandler('result.log')
    else:
        # 默认使用标准输出流
        handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s][%(module)s][%(threadName)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger



# 读入文本，并根据参数实现对数据的处理
# @return array
def readTXT(path, shape=None, norm=True):
    data = np.loadtxt(path)
    if shape:
        data = np.reshape(data, shape)
    return normalize(data) if norm else data


# 标准化函数，默认z-score
def normalize(data, ntype='zscore'):
    if ntype == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif ntype == 'log':
        return np.log(data)


# 解析Json文件
def readJson(path):
    with open(path) as f:
        res = json.load(f)
    return res


# 模型配置文件
# name:属性名
# group_name:属性前缀名，缺省值为None
# default:属性缺省值
# path:配置文件路径，默认为当前路径
def getProperties(name,group_name=None,default=None,path='config.json'):
    ret = default
    # 处理配置文件未读入的情况
    if not params.__contains__('config'):
        params['config'] = readJson(path)
    if group_name != None:
        ret = []
        # 处理属性未定义的情况
        try:
            group = params['config']['model'][group_name]
            for item in group:
                ret.append(item[name])
        except:
            ret = default
    else:
        # 返回读取的数据或缺省值
        if params['config']['model'].__contains__(name):
            ret = params['config']['model'][name]
    if ret == None and default == None:
        #params['log'].info("can't find property [%s] from the configurable
        #file and it has no default value specified,exit" % name)
        print('can\'t find property [%s] from the configurable file and it has no default value specified,exit' % name)
        exit(-1)
    return ret
    
    
# read image
# @return list
def readIMG(path, shape=None):
    png = Image.open(path)
    seq = png.getdata()
    ret = []
    for i in list(seq):
        for c in i:
            ret.append(c)
    return np.reshape(ret, shape) if shape else ret
    

# k:滤波核边长
# txt_input,image_input,image_truth应为[?,height,width,channels]形数据
# 随机划分场景
def sliceScene(txt_input, image_input, image_truth, sizeX=None,sizeY=None):
    if sizeX == None or sizeY == None:
        sizeX = params['batch_width']
        sizeY = params['batch_height']
    # 保证sizeX与sizeY不为None
    assert(sizeX != None and sizeY != None)
    # 保证源数据大小相同
    assert(txt_input.shape[1] == image_input.shape[1])
    assert(txt_input.shape[1] == image_truth.shape[1])
    assert(txt_input.shape[2] == image_input.shape[2])
    assert(txt_input.shape[2] == image_truth.shape[2])
    # 当前场景规模
    n = txt_input.shape[0]
    h = txt_input.shape[1]
    w = txt_input.shape[2]
    # 截取位置的左上角
    x = np.random.randint(0, w - sizeX + 1)
    y = np.random.randint(0, h - sizeY + 1)
    # 截取数据
    a = txt_input[0:n + 1, y:y + sizeY, x:x + sizeX, :]
    b = image_input[0:n + 1, y:y + sizeY, x:x + sizeX, :]
    c = image_truth[0:n + 1, y:y + sizeY, x:x + sizeX, :]
    return a, b, c


# 默认从场景中随机选择size份batch
# 其中ra=[65x65],rb=[65x65],rc=[65x65]
def next_batch(size, sizeX=None, sizeY=None):
    global params
    if sizeX == None or sizeY == None:
        sizeX = params['batch_width']
        sizeY = params['batch_height']
    # 保证sizeX与sizeY不为None
    assert(sizeX != None and sizeY != None)
    # 随机选择场景
    ids = np.random.randint(0, len(params['train_config']['scenes']), size)
    ra = []
    rb = []
    rc = []
    for i in ids:
        scene = params['train_config']['scenes'][i]
        if params['sceneCache'].__contains__(scene['name']):
            curScene = params['sceneCache'][scene['name']]
            a, b, c = sliceScene(curScene['txt_input'],
            curScene['image_input'], curScene['image_truth'], sizeX, sizeY)
        else:
            # 创建对应字典项
            curScene = {}
            spath = scene['path']
            jsonScene = readJson(params['train_data_dir'] + spath + '/config.json')
            curScene['width'] = jsonScene['attributes']['width']
            curScene['height'] = jsonScene['attributes']['height']
            curScene['channels'] = jsonScene['attributes']['channels']
            curScene['txt_input'] = readTXT(params['train_data_dir'] + spath + jsonScene['attributes']['txt_input'],
            (-1, curScene['height'], curScene['width'], curScene['channels']))
            curScene['image_input'] = readIMG(params['train_data_dir'] + spath + jsonScene['attributes']['image_input'],
            (-1, curScene['height'], curScene['width'], 3))
            curScene['image_truth'] = readIMG(params['train_data_dir'] + spath + jsonScene['attributes']['image_truth'],
            (-1, curScene['height'], curScene['width'], 3))
            if(len(params['sceneCache']) == params['maxCacheSize']):
                params['sceneCache'].popitem()
            # 向sceneCache中添加表项
            params['sceneCache'][scene['name']] = curScene
            # 场景随机划分
            a, b, c = sliceScene(curScene['txt_input'], curScene['image_input'],
            curScene['image_truth'], sizeX,sizeY)
        ra.extend(np.reshape(a, [-1, 1]))
        rb.extend(np.reshape(b, [-1, 1]))
        rc.extend(np.reshape(c, [-1, 1]))
    
    #ra = np.reshape(ra, [-1, sizeY, sizeX, curScene['channels']])
    ra = np.reshape(ra, [-1, curScene['channels']])
    rb = np.reshape(rb, [-1, 3])
    rc = np.reshape(rc, [-1, 3])
    # 数据返回
    return ra, rb, rc


# 用于预测效果：分割数据与图片，使之成为[65x65:default]大小的碎片，按顺序排列
# image_input [1,h,w,3]
# idx用于指定场景文件
def splitScene(batch_height=None,batch_width=None,idx=None):
    params['log'].info('spliting scene into batchs...')
    curScene = {}
    nScene = len(params['test_config']['scenes'])
    idx = int(np.random.randint(0,nScene) if idx == None else idx % nScene)
    scene = params['test_config']['scenes'][idx]
    params['pred_img_path'] = spath = scene['path']
    jsonScene = readJson(params['test_data_dir'] + spath + '/config.json')
    curScene['width'] = jsonScene['attributes']['width']
    curScene['height'] = jsonScene['attributes']['height']
    curScene['channels'] = jsonScene['attributes']['channels']
    curScene['txt_input'] = readTXT(params['test_data_dir'] + spath + jsonScene['attributes']['txt_input'],
    (-1, curScene['height'], curScene['width'], curScene['channels']))
    curScene['image_input'] = readIMG(params['test_data_dir'] + spath + jsonScene['attributes']['image_input'],
    (-1, curScene['height'], curScene['width'], 3))
    tbatchs = []
    ibatchs = []
    # 图像大小
    h = curScene['height']
    w = curScene['width']
    # batch大小
    # 默认不分割
    bh = h if batch_height == None else batch_height
    bw = w if batch_width == None else batch_width
    # 碎片数
    nh = math.ceil(float(h) / bh)
    nw = math.ceil(float(w) / bw)
    for y in range(nh):
        sy = min(bh * y, h - bh)
        for x in range(nw):
            sx = min(bw * x, w - bw)
            tbatchs.append(curScene['txt_input'][:,sy:sy + bh,sx:sx + bw,:])
            ibatchs.append(curScene['image_input'][:, sy:sy + bh, sx:sx + bw, :])
            # 添加txt部分
    #tbatchs = np.reshape(tbatchs,[-1,curScene['channels']])
    params['log'].info('scene splited.')
    return h,w,tbatchs,ibatchs


# 绘制图像
def onRender(queue=None, stream=None,save=True):
    params['log'].debug('onRender')
    if queue != None:
        idx = 0
        while True:
            params['log'].info('ready,waiting for receiving images...')
            image = queue.get()
            #params['log'].info('image get,ploting...')
            #plt.clf()
            #plt.imshow(image.astype(np.uint8))
            #plt.title('filtered image')
            # 保存图像到文件中
            if save:
                dstpath = params['test_data_dir'] + params['pred_img_path'] + '1' + '/pred-' + str(idx) + '.png'
                spm.toimage(image.astype(np.uint8),cmin=0,cmax=255).save(dstpath)
                params['log'].info('the image predicted was saved to: \'%s\'' % dstpath)
            #plt.show()
            #plt.close()
            idx+=1
    return

