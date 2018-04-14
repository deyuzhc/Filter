#!/usr/local/bin/python3
# encoding:utf-8
'''
创建各类的对象，并调用完成任务

'''

from prop import Prop
from iosched import IOsched
from mainproc import MainProc
from mainproc import MainFeed

import utils
import signal
import platform
import tensorflow as tf

if platform.system() == 'Windows':
    from irender import iRender
    import winsound as ws


def handler(sig, frame):
    logger = utils.getLogger()
    logger.info('closing...patient...')

    while not utils.getFlag('safeExit'):
        pass
    exit(1)


def main(argv=None):
    # 全局唯一日志句柄
    logger = utils.getLogger()
    # 是否可打断当前的操作
    utils.setFlag('safeExit', True)
    # 是否退出当前程序
    utils.setFlag('nowExit', False)

    # 处理SIGINT信号
    signal.signal(signal.SIGINT, handler)

    # 配置对象
    prop = Prop()
    logger.info(prop.toString())

    # 消息队列，用于中转训练误差信息
    msg_queue = utils.getQueue('msg')
    # 数据队列，用于中转训练数据
    data_queue = utils.getQueue('data')

    # 文件调度对象，加载数据数据
    sched = IOsched(prop, data_queue)
    sched.start()

    # 任务处理对象
    mainfeed = MainFeed(prop, data_queue)
    mainproc = MainProc(prop, mainfeed, msg_queue)
    mainproc.start()

    # 主线程等待终止消息或信号
    while not utils.getFlag('nowExit'):
        pass


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
