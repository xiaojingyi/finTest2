#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataMem.py
# Date: 2016 2016年05月07日 星期六 17时27分59秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"


import os, sys, time
import numpy as np
import threading
import gevent
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Data import Data
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
global g_is_running
global g_batch_data
global g_X, g_y, g_Xt, g_yt

# also, this is the base data class
class DataMem(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataMem init")
        self.config = config
        self.debug = config["debug"]
        super(DataMem, self).__init__(config)
        self.dcache_prefix = self.config["dcache_prefix"]
    
    def ignore(self, i, y):
        global g_X, g_y, g_Xt, g_yt

        g_X[i] = g_X[i] * 0.0 + 1.0
        g_y[i] = g_y[i] * 0.0 + y

    def placeAllData(self, X, y, Xt, yt, yt_=[]):
        global g_is_running, g_batch_data, g_data_use_len
        global g_X, g_y, g_Xt, g_yt, g_yt_

        g_X = np.array(X).astype(np.float32)
        g_y = np.array(y).astype(np.float32)
        g_X = self.pad(g_X)
        g_y = self.pad(g_y)

        fmean = self.dcache_prefix+"_mean.npy"
        if os.path.exists(fmean):
            mean = np.load(fmean)
        else:
            mean = g_X.mean(0)
            np.save(fmean, mean)
        fstd = self.dcache_prefix+"_std.npy"
        if os.path.exists(fstd):
            std = np.load(fstd)
        else:
            std = g_X.std(0)
            np.save(fstd, std)
            
        g_X = self.transform(g_X, mean, std)

        g_Xt = np.array(Xt).astype(np.float32)
        g_yt = np.array(yt).astype(np.float32)
        g_yt_ = np.array(yt_).astype(np.float32)
        g_Xt = self.pad(g_Xt)
        g_yt = self.pad(g_yt)
        g_yt_ = self.pad(g_yt_)
        g_Xt = self.transform(g_Xt, mean, std)

        self.data_len = len(g_y)
        self.data_idx = np.arange(self.data_len)

        self.initDataMem()
        return len(g_y), len(g_yt)

    def addDataSet(self):
        global g_is_running, g_batch_data, g_data_use_len

        print "dataset len: %d / %d" % (g_data_use_len, self.data_len)
        if g_data_use_len == self.data_len:
            return True

        after = int(g_data_use_len * (1 + self.data_growth))
        if after > self.data_len:
            g_data_use_len = self.data_len
        else:
            g_data_use_len = after
        return False # not reach the max length

    def stopData(self):
        global g_is_running, g_batch_data, g_data_use_len

        g_is_running = 0
        print "end:", not g_is_running
        #self.batch_loader.join(1)
        #print dir(self.batch_loader)
        #self.loader_thread.kill()
        self.batch_loader.join()
        while self.batch_loader.is_alive():
            print "alive"
            time.sleep(1)
        return True

    def initDataMem(self):
        global g_is_running, g_batch_data, g_data_use_len

        self.data_growth = self.config['data_growth']
        self.data_shape = self.config["data_shape"]
        self.batch_len = self.config["batch_len"]
        self.is_sim = self.config["is_sim"]

        g_is_running = 1
        g_data_use_len = self.config['data_init_len']
        g_batch_data = []

        self.data_lock = threading.Lock()
        self.test_batch = []

        # thread create batches
        self.batch_loader = threading.Thread(target=self.createBatches, args=([1]))
        self.batch_loader.start()
        #self.loader_thread = gevent.spawn(self.batch_loader.run())
        return

    def createBatches(self, param):
        global g_is_running, g_batch_data, g_data_use_len
        global g_X, g_y, g_Xt, g_yt

        assert self.batch_len < g_data_use_len

        last_i = 0
        while True:
            #print "gen data:", g_data_use_len
            check = g_is_running
            if not check:
                break
            idx_all = self.data_idx[0:g_data_use_len]

            idx = np.random.permutation(idx_all)[0:self.batch_len]
            X = g_X[idx]
            y = g_y[idx]
            if self.is_sim:
                idx = np.random.permutation(idx_all)[0:self.batch_len]
                X_ = g_X[idx]
                y_ = g_y[idx]

                X_final, y = self.simData(X, X_, y, y_)
            else:
                X_final = X

            self.data_lock.acquire()
            l = len(g_batch_data)
            self.data_lock.release()
            while l > 0:
                check = g_is_running
                #print "sleep 1", check
                if not check:
                    break
                time.sleep(0.001)
                self.data_lock.acquire()
                l = len(g_batch_data)
                self.data_lock.release()

            # do something
            self.data_lock.acquire()
            g_batch_data = [X_final, y]
            self.data_lock.release()

        return

    def loadTests(self):
        global g_X, g_y, g_Xt, g_yt, g_yt_
        if len(self.test_batch) <= 0:
            Xt = g_Xt
            yt = g_yt
            yt_ = g_yt_
            del g_Xt
            del g_yt
            del g_yt_

            if self.is_sim:
                Xt, yt = self.simData(Xt, Xt, yt, yt)
            self.test_batch = [Xt, yt, yt_]
        return self.test_batch

    def loadBaches(self):
        global g_is_running, g_batch_data, g_data_use_len

        self.data_lock.acquire()
        l = len(g_batch_data)
        self.data_lock.release()
        while l <= 0:
            #print "sleep 2"
            time.sleep(0.001)
            self.data_lock.acquire()
            l = len(g_batch_data)
            self.data_lock.release()

        data = g_batch_data

        self.data_lock.acquire()
        g_batch_data = []
        self.data_lock.release()
        return data

    def loadOrderBatchs(self, offset, length):
        global g_X, g_y, g_Xt, g_yt
        idx = np.arange(offset, offset+length)
        X = g_X[idx]
        y = g_y[idx]
        if self.is_sim:
            X_final, y = self.simData(X, X, y, y)
        else:
            X_final = X
        return [X_final, y]

    def useDataLen(self):
        global g_is_running, g_batch_data, g_data_use_len

        return g_data_use_len

    def isMax(self):
        global g_is_running, g_batch_data, g_data_use_len
        return g_data_use_len == self.data_len

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "data_init_len": 300,
            "cluster_number": 2,
            "data_growth": 0.1,
            "data_shape": (1, 2, 4), #TODO
            "batch_len": 100,
            "dcache_prefix": "",
            "is_sim": False,
            }
    t = DataMem(conf)
    X, y = sklearn.datasets.make_classification(
            n_samples=5000, n_features=30, n_redundant=5, n_informative=20,
            n_classes=2,
            n_clusters_per_class=3, hypercube=False, random_state=0
            )
    X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y, test_size=0.2)
    t.placeAllData(X, y, Xt, yt)
    for i in range (10):
        for j in range(2):
            batch = t.loadBaches()
            X = batch[0]
            y = batch[1]
            print X.shape
            print X.sum(), y.sum()
        batch = t.loadTests()
        X = batch[0]
        y = batch[1]
        print X.shape
        print X.sum(), y.sum()
        t.addDataSet()
    t.testPrint()
    t.stopData()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

