#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataStockRangeWeek.py
# Date: 2016 2016年05月29日 星期日 19时51分05秒
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

class DataStockRangeWeek(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataStockRangeWeek init")
        self.config = config
        self.debug = config["debug"]
        super(DataStockRangeWeek, self).__init__(config)
        print "DataStockRangeWeek"
    
    def initData(self):
        self.batch_len = self.config["batch_len"]
        self.is_sim = self.config["is_sim"]
        self.dcache_prefix = self.config["dcache_prefix"]
        self.data_growth = self.config['data_growth']
        self.dlen_use = self.config['data_init_len']
        return

    def placeAllData(self, X, y, Xt, yt, yt_=[]):
        print X.shape, Xt.shape
        self.X_base = X
        self.y = y
        self.Xt_base = Xt
        self.yt = yt
        self.yt_ = yt_
        self.Xlen = len(X)
        self.dlen = len(y)
        self.dtlen = len(yt)

        self.data_idx = self.genIdx(self.dlen_use)
        self.data_idx_t = self.genIdx(self.dtlen, True)

        f = self.dcache_prefix+"mean.npy"
        self.mean = np.load(f)[0:24+24*800]
        f = self.dcache_prefix+"std.npy"
        self.std = np.load(f)[0:24+24*800]

        #self.X_base = self.transform(self.X_base, self.mean, self.std)
        #self.Xt_base = self.transform(self.Xt_base, self.mean, self.std)

        self.nd_i = 0
        self.nd = []
        self.ndlen = 0

        return len(y), len(yt)

    def loadX(self, Xs, idx, blen):
        s = 24
        #l = 1508
        l = 800
        l_ = 2862
        X = np.zeros((blen, 1, 1, s + s*l))
        X_index = 0
        for one_i in idx:
            i = one_i / l_
            j = one_i % l_
            patten = Xs[i][0][0][j*s : j*s+s]
            trend = Xs[i][0][0][0 : s*l]
            X_i = np.concatenate( (patten, trend) )
            X[X_index][0][0] = X_i
            X_index += 1
        X = X.astype(np.float32)
        return X

    def loadBaches(self):
        t = time.time()
        if self.nd_i == 0:
            self.nd = np.random.permutation(self.data_idx)
            self.ndlen = len(self.nd)
        elif self.nd_i >= self.ndlen - self.batch_len:
            self.nd_i = 0
            self.nd = np.random.permutation(self.data_idx)
        t1 = time.time()
        idx = self.nd[self.nd_i : self.nd_i+self.batch_len]
        self.nd_i += self.batch_len
        X = self.loadX(self.X_base, idx, self.batch_len)
        X = self.transform(X, self.mean, self.std)
        y = self.y[idx]
        #print "1: " + str(t1 - t)
        return [X, y]

    def loadTests(self):
        batch_len = 1024 * 20
        idx = np.random.permutation(self.data_idx_t)[0:batch_len]
        idx.sort()
        X = self.loadX(self.Xt_base, idx, batch_len)
        X = self.transform(X, self.mean, self.std)
        y = self.yt[idx]
        y_ = self.yt_[idx]

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

    def genIdx(self, dlen, istest=False):
        if istest:
            idx = np.argwhere(self.yt.reshape(len(self.yt)) != 4)
        else:
            idx = np.argwhere(self.y.reshape(len(self.y)) != 4)
        #print idx.shape
        idx.shape = idx.shape[0]
        idx = np.random.permutation(idx)[0:dlen]
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
    t = DataStockRangeWeek(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

