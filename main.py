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
import tensorflow as tf
import platform

if platform.system() == 'Windows':
    from irender import iRender
    import winsound as ws


def main(argv=None):
    # 全局唯一日志句柄
    logger = utils.getLogger()

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
    mainproc = MainProc(prop, msg_queue)
    mainfeed = MainFeed(prop, data_queue)

    # 任务处理
    predict = mainproc.preprocess(prop, None)
    result = mainproc.process(mainfeed, predict)
    mainproc.postprocess(result)


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
