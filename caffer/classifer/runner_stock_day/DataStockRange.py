#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataStockRange.py
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

class DataStockRange(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataStockRange init")
        self.config = config
        self.debug = config["debug"]
        super(DataStockRange, self).__init__(config)
        print "DataStockRange"
    
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
        #idx = np.argwhere(self.y.reshape(self.dlen) != 0)
        #idx.shape = idx.shape[0]
        #self.data_idx = idx
        #self.train_idx = np.argwhere(self.y.reshape(self.dlen) != 4)
        #self.train_idx.shape = self.train_idx.shape[0]
        self.train_idx = np.arange(self.dlen)
        self.data_idx = self.genIdx(self.dlen_use)
        self.data_idx_t = np.arange(self.dtlen)
        #f = self.dcache_prefix+"_mean.npy"
        f = "./caffe/mean.npy"
        self.mean = np.load(f)[0: 4012]
        #self.mean = self.toStockMean(self.mean)
        #f = self.dcache_prefix+"_std.npy"
        f = "./caffe/std.npy"
        self.std = np.load(f)[0: 4012]
        #self.std = self.toStockMean(self.std)
        g_is_running = 1
        g_batch_data = []

        #self.data_lock = threading.Lock()
        #self.batch_loader = threading.Thread(target=self.createBatches, args=([1]))
        #self.batch_loader.start()
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
        #l_ = 2765
        l_ = 2862
        X = np.zeros((blen, 1, 1, 4012))
        X_index = 0
        for one_i in idx:
            i = one_i / l_
            j = one_i % l_
            patten = Xs[i][0][0][w+j*s : w+j*s+s]
            week = Xs[i][0][0][0 : w]
            trend = Xs[i][0][0][w : w+s*l]

            """
            if istrain:
                #tmp = trend.reshape((800, s))
                #t_idx = np.arange(800)
                #t_idx = np.random.permutation(t_idx)
                #trend = tmp[t_idx].reshape(800*s)
                if y[X_index] == self.balance_per[0]:
                    rand_n = np.random.randint(1000)
                    if rand_n > self.balance_per[1] * 1000:
                        patten = patten * 0 + self.placeholder
                        patten = self.norm(patten)
                        y[X_index] *= 0
            """

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
        X, y = self.loadX(self.X_base, idx, self.batch_len, y)

        """
        beta = 0.08
        noise = np.random.normal(0, X.std() * beta, X.shape)
        X += noise
        """

        """
        # make sim data
        idx_ = np.random.permutation(idx)
        y_ = self.y[idx_]
        X_, y_ = self.loadX(self.X_base, idx_, self.batch_len, y_)
        X_ = self.transform(X_, self.mean, self.std)

        X, y = self.simData(X, X_, y, y_)
        """
        return [X, y]

    def loadTests(self):
        batch_len = 1024 * 20
        idx = np.random.permutation(self.data_idx_t)[0:batch_len]
        idx.sort()
        y = self.yt[idx]
        X, y = self.loadX(self.Xt_base, idx, batch_len, y, False)
        """
        X, y = self.simData(X, X, y, y)
        """
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

    def loadBachesBak(self):
        global g_is_running, g_batch_data
        self.data_lock.acquire()
        l = len(g_batch_data)
        self.data_lock.release()
        while l <= 0:
            if not g_is_running:
                break
            #time.sleep(0.001)
            #print "get"
            self.data_lock.acquire()
            l = len(g_batch_data)
            self.data_lock.release()

        data = g_batch_data

        self.data_lock.acquire()
        g_batch_data = []
        self.data_lock.release()
        return data

    def createBatches(self, param): #TODO with new datas
        global g_is_running, g_batch_data
        nd_i = 0
        nd = []
        ndlen = 0
        while True:
            if nd_i == 0:
                nd = np.random.permutation(self.data_idx)
                ndlen = len(nd)
            elif nd_i >= nd_i - self.batch_len:
                nd_i = 0
                nd = np.random.permutation(self.data_idx)
            idx = nd[nd_i:nd_i+self.batch_len]
            nd_i += self.batch_len
            s = 5
            w = 7
            l = 800
            l_ = 2765
            X = np.zeros((self.batch_len, 1, 1, 4012))
            X_index = 0
            for one_i in idx:
                i = one_i / l_
                j = one_i % l_
                X_i = np.concatenate( (self.X_base[i][0][0][w+j*s : w+j*s+s], self.X_base[i][0][0][0 : w+s*l]) )
                X[X_index][0][0] = X_i
                X_index += 1
            X = X.astype(np.float32)
            X = self.transform(X, self.mean, self.std)
            y = self.y[idx]

            self.data_lock.acquire()
            l = len(g_batch_data)
            self.data_lock.release()
            while l > 0:
                check = g_is_running
                if not check:
                    break
                #time.sleep(0.001)
                #print "create"
                self.data_lock.acquire()
                l = len(g_batch_data)
                self.data_lock.release()

            if not g_is_running:
                break
            self.data_lock.acquire()
            g_batch_data = [X, y]
            self.data_lock.release()
        return

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = DataStockRange(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

