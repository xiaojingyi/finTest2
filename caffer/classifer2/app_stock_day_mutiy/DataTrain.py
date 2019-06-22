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
from data.StockSigle import StockSigle

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataTrain(StockSigle):
    def _commonConfigure(self):
        self.y_threshold = 0.0
        self.onelen = 4012
        self.X_noise_per = 0
        self.y_log_scale = 88
        self.use_id = True

    def _configure(self):
        self.is_train = True
        self.batch_size = 128
        self.tops = [
                (self.batch_size, 1, 1, self.onelen), 
                (self.batch_size, 20),
                (self.batch_size, 1),
                (self.batch_size, 1),
                ]

    def loadStockId(self):
        return np.load("./caffe/ids.npy")

    def loadStockDatas(self):
        return self.loadStockH5("./caffe/data_0.h5")

    def loadMeanStd(self):
        f = "./caffe/mean.npy"
        mean = np.load(f)[0: self.onelen]
        f = "./caffe/std.npy"
        std = np.load(f)[0: self.onelen]

        return mean, std

    def loadX(self, Xs, idx, blen, y, y_):
        s = 5
        w = 7
        l = 800
        l_ = 2862
        X, y = self._loadXFromIdx(
                Xs, idx, blen, [s, w, l, l_], 
                True, y, y_)
        return X, y

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

