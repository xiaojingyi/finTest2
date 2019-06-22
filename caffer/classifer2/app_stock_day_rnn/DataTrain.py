#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataTrain.py
# Date: 2016 Mon 01 Aug 2016 10:44:41 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../libs")
from data.StockSigleRnn import StockSigleRnn
import common

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataTrain(StockSigleRnn):
    def _commonConfigure(self):
        self.y_threshold = 0.0
        self.label_mean = 1
        self.onelen = 4 * (800 + 1)
        self.tlen = common.tlen
        self.stream_len = common.stream_len
        self.tops = [
                (self.tlen*self.stream_len, 1, 1, self.onelen), 
                (self.tlen*self.stream_len, 1, 1, 20), 
                (self.tlen*self.stream_len, 1),
                (self.tlen*self.stream_len, 1),
                (self.tlen, self.stream_len),
                ]

    def _configure(self):
        self.is_train = True

    def loadStockDatas(self):
        return self.loadStockH5("./caffe/data_0.h5")

    def loadMeanStd(self):
        f = "./caffe/mean.npy"
        mean = np.load(f)[0: self.onelen]
        f = "./caffe/std.npy"
        std = np.load(f)[0: self.onelen]

        return mean, std

def main():
    conf = {
            "debug": True,
            }
    t = DataTrain(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

