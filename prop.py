#!/bin/local/python3
# encoding:utf-8
'''
程序属性模块
'''
import utils
from cache import Cache
from shared import Shared
from singleton import Singleton


class Prop(metaclass=Singleton):
    def __init__(self):
        sd = Shared()
        self.__logger = sd.getLogger()
        self.__properties = {}
        self.__properties['CONFIG_FILE'] = 'config.json'
        self.__properties['mode'] = self.__getAttr('mode', 'train', self.__properties['CONFIG_FILE'])
        if self.__properties['mode'] == 'train':
            self.__properties['learning_rate'] = self.__getAttr('learning_rate', 1e-4, self.__properties['CONFIG_FILE'])
            self.__properties['max_round'] = self.__getAttr('max_round', 8e+3, self.__properties['CONFIG_FILE'])
            self.__properties['batch_n'] = self.__getAttr('batch_n', 1, self.__properties['CONFIG_FILE'])
            self.__properties['data_dir'] = self.__getAttr('train_data', 'data/train/',
                                                           self.__properties['CONFIG_FILE'])
            self.__properties['loss_func'] = self.__getAttr('loss_func', 'l1', self.__properties['CONFIG_FILE'])
        else:
            assert (self.__properties['mode'] == 'infer')
            self.__properties['batch_n'] = self.__getAttr('batch_n', 1, self.__properties['CONFIG_FILE'])
            if self.__properties['batch_n'] != 1:
                self.__logger.error(
                    'batch_num must be [1] in [predict] mode,change current value [%d] to [1]!' % self.__properties[
                        'batch_n'])
                self.__properties['batch_n'] = 1
            self.__properties['data_dir'] = self.__getAttr('test_data', 'data/test/', self.__properties['CONFIG_FILE'])
        self.__properties['conv_size'] = self.__getAttr('conv_size', 5, self.__properties['CONFIG_FILE'])
        self.__properties['graph_name'] = self.__getAttr('graph_name', 'model', self.__properties['CONFIG_FILE'])
        self.__properties['conf_input'] = self.__getAttr('conf_input', 'config.json', self.__properties['CONFIG_FILE'])
        self.__properties['ground_truth'] = self.__getAttr('ground_truth', 'truth.png',
                                                           self.__properties['CONFIG_FILE'])
        self.__properties['cache_size'] = self.__getAttr('cache_size', 800, self.__properties['CONFIG_FILE'])
        self.__properties['batch_h'] = self.__getAttr('batch_h', 50, self.__properties['CONFIG_FILE'])
        self.__properties['batch_w'] = self.__getAttr('batch_w', 50, self.__properties['CONFIG_FILE'])
        self.__properties['log_file'] = self.__getAttr('log_file', 'run.log', self.__properties['CONFIG_FILE'])
        self.__properties['sfeatures'] = self.__getAttr('sfeatures', 8, self.__properties['CONFIG_FILE'])
        self.__properties['ifeatures'] = self.__getAttr('ifeatures', 2, self.__properties['CONFIG_FILE'])
        assert (self.__properties['ifeatures'] == 2)
        self.__properties['optimizer'] = self.__getAttr('optimizer', 'Adam', self.__properties['CONFIG_FILE'])
        self.__properties['model_path'] = self.__getAttr('model_path', 'data/model/', self.__properties['CONFIG_FILE'])
        self.__properties['ckpt_name'] = self.__getAttr('ckpt_name', 'model', self.__properties['CONFIG_FILE'])
        self.__properties['cnn_name'] = self.__getAttr('cnn_name', ['global', 'caustic'],
                                                       self.__properties['CONFIG_FILE'])
        self.__properties['active_func'] = self.__getAttr('active_func', ['relu', 'sigmoid', 'relu', 'sigmoid'],
                                                          self.__properties['CONFIG_FILE'])
        self.__properties['weights_shape'] = self.__getAttr('weights_shape', [8, 100, 100, 100, 121],
                                                            self.__properties['CONFIG_FILE'])
        assert (len(self.__properties['active_func']) + 1 == len(self.__properties['weights_shape']))
        # 图像通道，默认为3
        self.__properties['cols'] = self.__getAttr('cols', 3, self.__properties['CONFIG_FILE'])
        self.__properties['plot_high'] = self.__getAttr('plot_high', 300, self.__properties['CONFIG_FILE'])
        self.__properties['plot_width'] = self.__getAttr('plot_width', 100, self.__properties['CONFIG_FILE'])
        self.__properties['session'] = None
        self.__properties['cache'] = Cache(self.queryAttr('cache_size'))

    ############################
    #########外部函数###########
    ############################

    def needTrain(self):
        return self.__properties['mode'] == 'train'

    '''
    @about
        外部接口，返回查询的属性值
    @param
        name:属性名
    '''

    def queryAttr(self, name):
        return self.__properties[name]

    '''
    @about
        外部接口，更新属性值
    @param
        name:属性名
        value:属性值
    '''

    def updateAttr(self, name, value):
        # assert(False)
        self.__setAttr(name, value)

    '''
    @about
        外部接口，以字符串输出所有的属性及其值
    @param
        None
    '''

    def toString(self):
        result = '\n\n\t\t[Properties]:\n'
        result += '# [mode]:\t\t[\'' + self.__properties['mode'] + '\']\n'
        if self.__properties['mode'] == 'train':
            result += '# [max_round]:\t\t[' + str(self.__properties['max_round']) + ']\n'
            result += '# [learning_rate]:\t[' + str(self.__properties['learning_rate']) + ']\n'
            result += '# [loss_func]:\t\t[\'' + str(self.__properties['loss_func']) + '\']\n'
            result += '# [optimizer]:\t\t[\'' + str(self.__properties['optimizer']) + '\']\n'
        else:
            assert (self.__properties['mode'] == 'infer')
        result += '# [batch_num]:\t\t[' + str(self.__properties['batch_n']) + ']\n'
        result += '# [batch_height]:\t[' + str(self.__properties['batch_h']) + ']\n'
        result += '# [batch_width]:\t[' + str(self.__properties['batch_w']) + ']\n'
        result += '# [conv_size]:\t\t[' + str(self.__properties['conv_size']) + ']\n'
        result += '# [sfeatures]:\t\t[' + str(self.__properties['sfeatures']) + ']\n'
        result += '# [ifeatures]:\t\t[' + str(self.__properties['ifeatures']) + ']\n'
        result += '# [plot_high]:\t\t[' + str(self.__properties['plot_high']) + ']\n'
        result += '# [plot_width]:\t\t[' + str(self.__properties['plot_width']) + ']\n'
        result += '# [model_path]:\t\t[\'' + str(self.__properties['model_path']) + '\']\n'
        result += '# [cnn_name]:\t\t' + str(self.__properties['cnn_name']) + '\n'
        result += '# [ckpt_name]:\t\t[\'' + str(self.__properties['ckpt_name']) + '\']\n'
        result += '# [data_dir]:\t\t[\'' + str(self.__properties['data_dir']) + '\']\n'
        result += '# [ground_truth]:\t[\'' + str(self.__properties['ground_truth']) + '\']\n'
        result += '# [log_file]:\t\t[\'' + str(self.__properties['log_file']) + '\']\n'
        result += '# [conf_input]:\t\t[\'' + str(self.__properties['conf_input']) + '\']\n'
        result += '# [cache_size]:\t\t[' + str(self.__properties['cache_size']) + 'M]\n'
        result += '# [weights_shape]:\t' + str(self.__properties['weights_shape']) + '\n'
        result += '# [active_func]:\t' + str(self.__properties['active_func']) + '\n'
        return result

    ############################
    #########内部函数###########
    ############################

    '''
    @about
        内部接口，返回查询的属性值
    @param
        name:   属性名
        default:缺省值
    '''

    def __getAttr(self, name, default, filename):
        result = utils.getJsonAttr(name, default, filename)
        return result

    '''
    @about
        内部接口，更新属性值
    @param
        name:   属性名
        value:缺省值
    '''

    def __setAttr(self, name, value):
        self.__properties[name] = value
