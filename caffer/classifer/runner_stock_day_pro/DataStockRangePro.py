#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataStockRangePro.py
# Date: 2016 2016年05月21日 星期六 17时13分45秒
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
global g_is_running
global g_batch_data

class DataStockRangePro(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataStockRangePro init")
        self.config = config
        self.debug = config["debug"]
        super(DataStockRangePro, self).__init__(config)
        print "DataStockRangePro"
    
    def initData(self):
        self.batch_len = self.config["batch_len"]
        self.is_sim = self.config["is_sim"]
        self.dcache_prefix = self.config["dcache_prefix"]
        self.data_growth = self.config['data_growth']
        self.dlen_use = self.config['data_init_len']
        return

    def placeAllData(self, X, y, Xt, yt, yt_=[]):
        global g_is_running, g_batch_data
        print X.shape, Xt.shape
        self.X_base = X
        self.y = y
        self.balance_per = self.balancePer(y)
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
        self.mean = np.load(f)[0: 4012]
        f = "./caffe/std.npy"
        self.std = np.load(f)[0: 4012]
        g_is_running = 1
        g_batch_data = []

        self.nd_i = 0
        self.nd = []
        self.ndlen = 0

        return len(y), len(yt)

    def toStockMean(self, X):
        shape = X.shape
        #print shape
        X = X.reshape(shape[-1])
        week = X[0:7].tolist()
        first = X[7:7+5].tolist()
        main = X[7+5:len(X)].tolist()
        final = first
        final.extend(week)
        final.extend(main)
        return np.array(final).reshape(shape)

    def loadX(self, Xs, idx, blen, y, istrain=True):
        s = 5
        w = 7
        l = 800
        l_ = 2862
        X = self._loadX(Xs, idx, blen, [s, w, l, l_], is_test=istrain)
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
        X, y = self.loadX(self.X_base, idx, self.batch_len, y)

        return [X, y]

    def loadTests(self):
        batch_len = 1024 * 20
        idx = np.random.permutation(self.data_idx_t)[0:batch_len]
        idx.sort()
        y = self.yt[idx]
        X, y = self.loadX(self.Xt_base, idx, batch_len, y, False)
        y_ = self.yt_[idx]

        return [X, y, y_]

    def stopData(self):
        global g_is_running, g_batch_data

        g_is_running = 0
        print "end:", not g_is_running
        self.batch_loader.join()
        while self.batch_loader.is_alive():
            print "alive"
            time.sleep(1)
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
    t = DataStockRangePro(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

