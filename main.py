#!/usr/local/bin/python3
# encoding:utf-8
'''
创建各类的对象，并调用完成任务

'''

from prop import Prop
from shared import Shared
from iosched import IOsched
from mainproc import MainProc
from mainproc import MainFeed

import signal
import platform
import tensorflow as tf

if platform.system() == 'Windows':
    from irender import iRender
    import winsound as ws


def handler(sig, frame):
    sd = Shared()
    logger = sd.getLogger()
    logger.debug("\r\r\r\r\r")
    print('...closing...patient...')

    while sd.getFlag('safeExit') != 0:
        pass
    sd.setFlag('nowExit', True)


def main(argv=None):
    sd = Shared()
    # 全局唯一日志句柄
    logger = sd.getLogger()
    # 当前操作是否可打断
    sd.setFlag('safeExit', 0)
    # 当前是否可退出
    sd.setFlag('nowExit', False)

    # 处理SIGINT信号
    signal.signal(signal.SIGINT, handler)

    # 配置对象
    prop = Prop()
    logger.info(prop.toString())

    # 消息队列，用于中转训练误差信息
    msg_queue = sd.getQueue('msg')
    # 数据队列，用于中转训练数据
    data_queue = sd.getQueue('data')

    # 文件调度对象，加载数据数据
    sched = IOsched(prop, data_queue)
    sched.start()

    # 任务处理对象
    mainfeed = MainFeed(prop, data_queue)
    mainproc = MainProc(prop, mainfeed, msg_queue)
    mainproc.start()

    # 走势图绘制
    if platform.system() == 'Windows':
        render = iRender(msg_queue, prop)

    # 主线程等待终止消息或信号
    while not sd.getFlag('nowExit'):
        if platform.system() == 'Windows':
            render.process()
        else:
            pass


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
