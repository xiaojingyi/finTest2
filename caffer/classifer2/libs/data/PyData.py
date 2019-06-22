#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: PyData.py
# Date: 2016 Mon 01 Aug 2016 12:24:47 AM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import caffe
import numpy as np
sys.path.append("/datas/lib/py")
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class PyData(caffe.Layer):

    ############# rewrite start #############
    def _commonConfigure(self):
        pass

    def _configure(self):
        pass

    def loadData(self):
        pass

    def batch(self):
        return

    ############# rewrite end #############

    def forward(self, bottom, top):
        assert len(top) == self.top_len

        t = time.time()
        batch_data = self.batch()
        for i, one in enumerate(self.tops):
            shape = top[i].data.shape
            top[i].data[...] = batch_data[i].reshape(shape)
        t_ = time.time()

    def configure(self):
        self.batch_size = 128
        self.tops = [
                (self.batch_size, 1, 1, 1000), 
                (self.batch_size, 20),
                ]
        self.label_padding = []
        self.X_noise_per = 0
        self.y_log_scale = 66
        self.up_threshold = self.sell_threshold = 0
        self.y_threshold = 0
        self.idx_skip = 0
        self.use_id = False # # unfinished TODO
        self.use_seperate_idx = 0.3
        self.test_n = 2 # n * 10240
        self._commonConfigure()
        self._configure()
        self.top_len = len(self.tops)

    def setup(self, bottom, top):
        # configure
        self.configure()

        # init the datas
        self.initData()

        # reshape the tops( after config )
        for i, one in enumerate(self.tops):
            top[i].reshape(*one)

    def initData(self):
        self.loadData()
        return

    # util methods
    def scale(self, X, scale=255):
        return X * scale

    def transform(self, X, mean, std, scale=255):
        return self.scale((X - mean) / std, scale)

    ############## ignore from here ##############
    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

