#!/bin/python3
# encoding:utf-8

'''
单例模式元类
'''


class Singleton(type):

    def __init__(cls, name, bases, attrs):
        super(Singleton, cls).__init__(name, bases, attrs)
        cls.__instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__call__(*args, **kwargs)

        return cls.__instance
