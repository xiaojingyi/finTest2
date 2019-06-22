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
        self.y_threshold = 0.008
        nlen = 10
        self.onelen = 800 * nlen + nlen + 7
        self.nlen = nlen
        #self.label_padding = [0, 0, 5]
        self.y_log_scale = 88
        self.use_seperate_idx = 0.66

    def _configure(self):
        self.is_train = True
        self.batch_size = 512
        self.tops = [
                (self.batch_size, 1, 1, self.onelen*3), 
                (self.batch_size, 20),
                (self.batch_size, 1),
                (self.batch_size, 1),
                (self.batch_size, 1), # the sim
                ]

    def batchTransform(self, res):
        res.append(np.random.randint(2, size=(self.batch_size, 1)))
        #res.append(np.zeros((self.batch_size, 1)))
        return res

    def loadStockDatas(self):
        return self.loadStockH5("./caffe/data_0.h5")

    def loadMeanStd(self):
        f = "./caffe/mean.npy"
        mean = np.load(f)
        f = "./caffe/std.npy"
        std = np.load(f)
        mean = mean[:, 0:self.onelen]
        std = std[:, 0:self.onelen]

        return mean, std

    def loadX(self, Xs, idx, blen, y, y_):
        s = self.nlen
        w = 7
        l = 800
        l_ = 2862
        X = None
        for i in range(Xs.shape[1]):
            X_base = Xs[:, i]
            X_base.shape = (X_base.shape[0], 1, 1, X_base.shape[1])
            if i == 0:
                Xt, y = self._loadXFromIdx(
                        X_base, idx, blen, [s, w, l, l_], 
                        True, y, y_, meanstd_index=i)
                X = Xt.copy()
            else:
                Xt, _ = self._loadXFromIdx(
                        X_base, idx, blen, [s, w, l, l_], 
                        False, y, y_, meanstd_index=i)
                X = np.concatenate((X, Xt), axis=3)
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

