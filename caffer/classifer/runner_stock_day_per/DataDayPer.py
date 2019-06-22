#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataDayPer.py
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

class DataDayPer(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataDayPer init")
        self.config = config
        self.debug = config["debug"]
        super(DataDayPer, self).__init__(config)
        print "DataDayPer"
    
    def initData(self):
        self.batch_len = self.config["batch_len"]
        self.is_sim = self.config["is_sim"]
        self.dcache_prefix = self.config["dcache_prefix"]
        self.data_growth = self.config['data_growth']
        self.dlen_use = self.config['data_init_len']
        return

    def placeAllData(self, X, y, Xt, yt, yt_=[], y_=[]):
        global g_is_running, g_batch_data
        print X.shape, Xt.shape

        self.X_base = X
        y_.shape = y.shape
        self.y = y
        y_ = abs(y_)
        self.y_ = (y_ - y_.min()) / (y_.max() - y_.min())
        #self.y_ = abs(abs(y_-y_.mean()) - y_.std())
        #self.y_ = (1 - (self.y_ - self.y_.min()) / (self.y_.max() - self.y_.min())) / 1.

        self.Xt_base = Xt
        yt_.shape = yt.shape
        self.yt = yt
        self.yt_ = yt_

        self.Xlen = len(X)
        self.dlen = len(y)
        self.dtlen = len(yt)

        #self.train_idx = np.argwhere(self.y.reshape(self.dlen) != 4)
        #self.train_idx.shape = self.train_idx.shape[0]
        self.train_idx = np.arange(self.dlen)
        self.data_idx = self.genIdx(self.dlen_use)
        self.data_idx_t = np.arange(self.dtlen)
        self.Xi_len = 5
        self.num = 2862
        self.one_len = self.Xi_len + 7 + self.num * self.Xi_len
        f = "./caffe/mean.npy"
        self.mean = np.load(f)[0: self.one_len]
        self.mean = np.concatenate((self.mean[0:7+self.Xi_len-1], self.mean[7+self.Xi_len:]))
        f = "./caffe/std.npy"
        self.std = np.load(f)[0: self.one_len]
        self.std = np.concatenate((self.std[0:7+self.Xi_len-1], self.std[7+self.Xi_len:]))
        g_is_running = 1
        g_batch_data = []

        self.nd_i = 0
        self.nd = []
        self.ndlen = 0

        return len(y), len(yt)

    def loadX(self, Xs, idx, blen, y, istrain=True):
        s = self.Xi_len
        w = 7
        l = self.num
        l_ = 2862
        X = np.zeros((blen, 1, 1, self.one_len-1))
        X_index = 0
        t_idx = np.arange(l)
        for one_i in idx:
            i = one_i / l_
            j = one_i % l_
            patten = Xs[i][0][0][w+j*s : w+j*s+s - 1]
            week = Xs[i][0][0][0 : w]
            trend = Xs[i][0][0][w : w+s*l]

            if istrain:
                tmp = trend.reshape((l, s))
                t_idx = np.random.permutation(t_idx)
                trend = tmp[t_idx].reshape(l*s)

            if np.random.randint(1000) < 50 and istrain:
                patten = patten * 0 + self.placeholder
                y[X_index] = 0

            X_i = np.concatenate( (patten, week, trend) )
            X[X_index][0][0] = X_i
            X_index += 1
        X = X.astype(np.float32)
        return X, y

    def randDrop(self, X, width=4, num=800, weeklen=7):
        choose = np.random.choice(800, 100)
        for one in choose:
            index = width + weeklen + one * width
            X[index: index+width] *= 0
            X[index: index+width] += self.placeholder
        return X

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
        y_ = abs(self.y_[idx])
        X, y = self.loadX(self.X_base, idx, self.batch_len, y)
        X = self.transform(X, self.mean, self.std)
        #X = np.concatenate((X, y_), axis=3)

        # make sim data
        #idx_ = np.random.permutation(idx)
        #X_ = self.loadX(self.X_base, idx_, self.batch_len)
        #X_ = self.transform(X_, self.mean, self.std)
        #y_ = self.y[idx_]
        #X, y = self.simData(X, X_, y, y_)

        return [X, y]

    def loadTests(self):
        batch_len = 1024 * 20
        idx = np.random.permutation(self.data_idx_t)[0:batch_len]
        idx.sort()

        y = self.yt[idx]
        y_ = self.yt_[idx]
        X, y = self.loadX(self.Xt_base, idx, batch_len, y, False)
        X = self.transform(X, self.mean, self.std)
        #X = np.concatenate((X, y_), axis=3)
        #X = np.concatenate((X, abs(y_)), axis=3)

        #X, y = self.simData(X, X, y, y)

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
    t = DataDayPer(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

