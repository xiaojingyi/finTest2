#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataStockDays.py
# Date: 2016 Sat 09 Jul 2016 12:01:47 AM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
from Data import Data
import numpy as np
import threading
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataStockDays(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataStockDays init")
        self.config = config
        self.debug = config["debug"]
        super(DataStockDays, self).__init__(config)
        print "DataStockDays"
        self.one_dim = 4 * 10
    
    def initData(self):
        self.batch_len = self.config["batch_len"]
        self.is_sim = self.config["is_sim"]
        self.dcache_prefix = self.config["dcache_prefix"]
        self.data_growth = self.config['data_growth']
        self.dlen_use = self.config['data_init_len']
        return

    def normY_(self, y_):
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

        per = abs(y_pos.mean() / y_neg.mean())
        if per < 1:
            y_[ np.where(y_ < 0) ] *= per
        else:
            y_[ np.where(y_ < 0) ] /= per

        per = len(y_pos) / len(y_neg)
        print len(y_pos), len(y_neg)
        if per < 1:
            y_[ np.where(y_ < 0) ] *= per
        else:
            y_[ np.where(y_ < 0) ] /= per

        y_neg = y_[ np.where(y_ < 0) ]
        print y_pos.mean(), y_neg.mean()
        return y_

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
        f = "./caffe/mean.npy"
        self.mean = np.load(f)[0: self.one_dim + 800 * self.one_dim]
        f = "./caffe/std.npy"
        self.std = np.load(f)[0: self.one_dim + 800 * self.one_dim]

        self.nd_i = 0
        self.nd = []
        self.ndlen = 0

        return len(y), len(yt)

    def loadX(self, Xs, idx, blen, y, y_, is_test=False):
        s = self.one_dim
        w = 0
        l = 800
        l_ = 2862
        alpha = 0.001
        X = np.zeros((blen, 1, 1, s*l+s+w))
        y_profit = np.zeros((blen, 1, 1, 20))
        y_profit[:, :, :, 0:4] += alpha
        X_index = 0
        for one_i in idx:
            i = one_i / l_
            j = one_i % l_
            patten = Xs[i][0][0][w+j*s : w+j*s+s]
            week = Xs[i][0][0][0 : w]
            trend = Xs[i][0][0][w : w+s*l]

            X_i = np.concatenate( (patten, week, trend) )
            X[X_index][0][0] = X_i

            X_index += 1
        X = X.astype(np.float32)
        X = self.transform(X, self.mean, self.std, 255)
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
    t = DataStockDays(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

