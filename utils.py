#!/bin/local/python3
# encoding:utf-8
'''
logger模块
文件存取
图像显示
'''

import sys
import cv2
import json
import numpy as np
import scipy.misc as spm

from PIL import Image
from hashlib import md5
from shared import Shared
from tensorflow.python.framework.ops import Tensor

'''
@about
    读取并解析json文件，返回指定属性的值
    读入文件后将其放入内存中，到程序结束后再释放
@param
    name:   属性名
    default:缺省值
'''


def getJsonAttr(name, default, filename):
    global result
    try:
        with open(filename) as f:
            result = json.load(f)
        return result[name]
    except:
        return default


'''
@about
    读入json文件
@param
    filename:文件名
@return
    json数据
'''


def readJson(filename):
    with open(filename) as f:
        ret = json.load(f)
    return ret


'''
@about
    写出json到文件
@param
    content:dict类型
    filename:目标文件
@return
    None
'''


def writeJson(content, filename):
    isinstance(content, dict)
    with open(filename, 'w') as f:
        json.dump(content, f)
    return


'''
@about
    读图片文件，像素各通道值位于[0,255]之间
@param
    filename:图像路径
@return
    RGB于[0,255]之间，图像[1,h,w,c]
'''


def readIMG(filename):
    image = Image.open(filename)
    result = np.array(image)
    ih = result.shape[0]
    iw = result.shape[1]
    ic = result.shape[2]
    result = np.reshape(result, [1, ih, iw, ic])
    return result / 255
    # return result


'''
@about
    读纯由数字组成的文本文件
@param
    filename:文本路径
    shape:自定义shape
@return
    返回读入的文本
'''


def readTXT(filename, shape=None):
    shared = Shared()
    logger = shared.getLogger()
    logger.debug('loading txt:%s...' % filename)
    data = np.loadtxt(filename)
    if shape:
        data = np.reshape(data, shape)
    logger.debug('txt loaded.')
    return data


'''
@about
    获取指定对象大小，
    仅保证正确支持Python内置或numpy对象
    递归实现
@param
    目标对象
@return
    整数，以B为单位的大小
'''


def getSize(item):
    size = 0
    if isinstance(item, dict):
        for name in item.keys():
            size += getSize(item[name])
    elif isinstance(item, np.ndarray):
        size += item.nbytes
    elif isinstance(item, list):
        for itm in item:
            size += getSize(itm)
    else:
        assert (not isinstance(item, Tensor))
        size += sys.getsizeof(item)
    return int(size)


'''
@about
    从numpy数组中截取指定大小
@param
    src:np.ndarray
    begin:与src同维
    size:与begin同维
@return
    指定大小的截取值
'''


def slice(src, begin, size):
    assert (isinstance(src, np.ndarray))
    assert (isinstance(begin, list))
    assert (isinstance(size, list))
    assert (len(src.shape) == len(begin))
    assert (len(src.shape) == len(size))
    sn = begin[0]
    sh = begin[1]
    sw = begin[2]
    sc = begin[3]
    dn = size[0]
    dh = size[1]
    dw = size[2]
    dc = size[3]
    result = src[sn:sn + dn, sh:sh + dh, sw:sw + dw, sc:sc + dc]
    return result


'''
@about
    给定多张[h,w,3]的RGB图像
    计算各张图像的梯度并返回
@param
    input:
@return
    magnitued:[n,h,w,1]
'''


def getMagnitude(input):
    n = input.shape[0]
    h = input.shape[1]
    w = input.shape[2]
    mag = np.ones([n, h, w, 1])
    lum = getLuminance(input)
    for i in range(n):
        layer = np.reshape(lum[i], [h, w])
        dx = cv2.Sobel(layer, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(layer, cv2.CV_64F, 0, 1, ksize=5)
        cm = cv2.magnitude(dx, dy)
        mag[i] = np.reshape(cm, [h, w, 1])
    return mag


'''
@about
    输入RGB三通道图像[n,h,w,3]，返回对应亮度值[n,h,w,1]
@param
    input:输入三通道图像[n,h,w,3]
@return
    [n,h,w,1]
'''


def getLuminance(input):
    assert (len(input.shape) == 4)
    n = input.shape[0]
    h = input.shape[1]
    w = input.shape[2]
    trans = np.array([0.299, 0.587, 0.114])
    result = np.sum(input * trans, -1)
    result = np.reshape(result, [n, h, w, 1])
    return result


'''
@about
    返回文件行数
@param
    filename:文件名
@return
    文件的行数
'''


def getLines(filename):
    lines = -1
    for lines, line in enumerate(open(filename)):
        pass
    lines += 1
    return lines


'''
@about
    md5检验
@param
    文件名
@return
    md5校验和
'''


def md5sum(filename):
    m = md5()
    file = open(filename, 'rb')
    line = file.readline()
    while line:
        m.update(line)
        line = file.readline()
    file.close()
    lower = m.hexdigest()
    return lower.upper()


'''
@about
    保存图像,各通道亮度值[0,255]，然后保存
@param
    data:       源数据，三维或四维
    filename:   文件路径
@return
    None
'''


def saveImage(data, filename):
    sd = Shared()
    sd.incFlag('safeExit')
    logger = sd.getLogger()
    src = (data * 255).astype(np.uint8)
    if len(src.shape) == 4:
        assert (src.shape[0]) == 1
        h = src.shape[1]
        w = src.shape[2]
        c = src.shape[3]
        src = np.reshape(src, [h, w, c])
    else:
        assert (len(src.shape) == 3)
    spm.toimage(src).save(filename)
    logger.debug('image saved to \'%s\'' % filename)
    sd.decFlag('safeExit')


'''
@about
    显示图片，仅windows可用
@param
    *img:变长参数，所有将要显示的图像
    数组中每个元素均为4维
    title:总标题
@return
    None
'''


def displayImage(*img, title='result'):
    from matplotlib import pyplot as plt
    assert (isinstance(i, np.ndarray) for i in img)
    plt.title(title)
    sum = len(img)
    nh = int(np.sqrt(sum))
    nw = int(np.ceil(sum / nh))
    for i in range(sum):
        assert (len(img[i].shape) == 4)
        image = np.reshape(img[i], [img[i].shape[1], img[i].shape[2], img[i].shape[3]])
        plt.subplot(nh, nw, i + 1)
        plt.title(str(i))
        # plt.imshow(image)
        plt.imshow(image.astype(np.uint8))
    plt.show()
