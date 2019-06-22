#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataStockDaysWithProfit.py
# Date: 2016 2016年07月03日 星期日 17时59分44秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
from Data import Data
import numpy as np
#import cudamat
import threading
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataStockDaysWithProfit(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataStockDaysWithProfit init")
        self.config = config
        self.debug = config["debug"]
        super(DataStockDaysWithProfit, self).__init__(config)
        print "DataStockDaysWithProfit"
    
    def initData(self):
        self.batch_len = self.config["batch_len"]
        self.is_sim = self.config["is_sim"]
        self.dcache_prefix = self.config["dcache_prefix"]
        self.data_growth = self.config['data_growth']
        self.dlen_use = self.config['data_init_len']
        return

    def normY_(self, y_):
        # transform
        scale = 20
        y_ *= scale
        y_[ np.where(y_ > 0) ] = np.log(y_[ np.where(y_ > 0) ] + 1)
        y_[ np.where(y_ < 0) ] = -np.log(abs(y_[ np.where(y_ < 0) ]) + 1)

        # to 0-1
        y_pos = y_[ np.where(y_ > 0) ]
        max_pos = y_pos.max()
        y_[ np.where(y_ > 0) ] /= max_pos
        y_pos = y_[ np.where(y_ > 0) ]
        print y_pos.max(), y_pos.min()

        y_neg = y_[ np.where(y_ < 0) ]
        min_neg = y_neg.min()
        y_[ np.where(y_ < 0) ] /= abs(min_neg)
        y_neg = y_[ np.where(y_ < 0) ]
        print y_neg.max(), y_neg.min()

        # balance the pos and neg
        y_pos = y_[ np.where(y_ > 0) ]
        y_neg = y_[ np.where(y_ < 0) ]
        per = abs(y_pos.mean() * 1.0 / y_neg.mean())
        if per < 1:
            y_[ np.where(y_ < 0) ] *= per
        else:
            y_[ np.where(y_ > 0) ] /= per

        per = len(y_pos) * 1.0 / len(y_neg)
        if per < 1:
            y_[ np.where(y_ < 0) ] *= per
        else:
            y_[ np.where(y_ > 0) ] /= per

        y_pos = y_[ np.where(y_ > 0) ]
        y_neg = y_[ np.where(y_ < 0) ]
        print y_pos.max(), y_pos.min()
        print y_neg.max(), y_neg.min()
        print y_pos.mean(), y_neg.mean()

        self.alpha = 1.0
        self.label_mean = float(y_pos.mean())
        return y_ * self.alpha

    def placeAllData(self, X, y, y_, Xt, yt, yt_=[]):
        global g_is_running, g_batch_data
        print X.shape, Xt.shape
        self.X_base = X
        self.y = y
        y_ = self.normY_(y_)
        self.y_ = y_
        self.Xt_base = Xt
        self.yt = yt
        self.yt_ = yt_
        self.Xlen = len(X)
        self.dlen = len(y)
        self.dtlen = len(yt)
        self.train_idx = np.arange(self.dlen)
        self.data_idx = self.genIdx(self.dlen_use)
        self.data_idx_t = np.arange(self.dtlen)
        self.one_len = 40
        self.trend_len = 800
        f = "./caffe/mean.npy"
        self.mean = np.load(f)[0: self.one_len + self.one_len * self.trend_len]
        #self.mean.shape = (1, 1, self.one_len * (1 + self.trend_len))
        f = "./caffe/std.npy"
        self.std = np.load(f)[0: self.one_len + self.one_len * self.trend_len]
        #self.std.shape = (1, 1, self.one_len * (1 + self.trend_len))

        self.nd_i = 0
        self.nd = []
        self.ndlen = 0
        self.bzeros = np.array([])
        return len(y), len(yt)

    def loadX(self, Xs, idx, blen, y, y_, is_test=False):
        s = self.one_len
        w = 0
        l = 800
        l_ = 2862
        X = self._loadX(Xs, idx, blen, [s, w, l, l_], y, y_, is_test)
        return X, y

    def loadBaches(self):
        if self.nd_i == 0:
            self.nd = np.random.permutation(self.data_idx)
            self.ndlen = len(self.nd)
        elif self.nd_i >= self.ndlen - self.batch_len:
            self.nd_i = 0
            self.nd = np.random.permutation(self.data_idx)
        idx = self.nd[self.nd_i : self.nd_i+self.batch_len]
        self.nd_i += self.batch_len

        y = self.y[idx]
        y_ = self.y_[idx]
        X, y = self.loadX(self.X_base, idx, self.batch_len, y, y_)

        return [X, y]

    def loadTests(self):
        batch_len = 1024 * 20
        idx = np.random.permutation(self.data_idx_t)[0:batch_len]
        idx.sort()
        y = self.yt[idx]
        y_ = self.yt_[idx]
        X, y = self.loadX(self.Xt_base, idx, batch_len, y, y_, is_test=True)

        return [X, y, y_]

    def stopData(self):
        return True

    def addDataSet(self):
        if self.dlen_use == self.dlen:
            return True

        after = int(self.dlen_use * (1 + self.data_growth))
        if self.dlen_use > self.dlen:
            self.dlen_use = self.dlen
        else:
            self.dlen_use = after
        self.data_idx = self.genIdx(self.dlen_use)
        return False

    def isMax(self):
        return self.dlen == self.dlen_use

    def useDataLen(self):
        return self.dlen_use

    def genIdx(self, dlen):
        idx = np.random.permutation(self.train_idx)[0:dlen]
        return idx

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = DataStockDaysWithProfit(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

