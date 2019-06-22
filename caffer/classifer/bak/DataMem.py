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
import sklearn
import sklearn.datasets
import sklearn.linear_model

sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

# also, this is the base data class
class DataMem(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataMem init")
        self.config = config
        self.debug = config["debug"]
        #super(DataMem, self).__init__(config)
    
    def placeAllData(self, X, y, Xt, yt):
        self.data_len = len(X)
        self.data_idx = np.arange(self.data_len)
        self.X = np.array(X).astype(np.float32)
        self.y = np.array(y).astype(np.float32)
        self.Xt = np.array(Xt).astype(np.float32)
        self.yt = np.array(yt).astype(np.float32)
        self.initData()

    def addDataSet(self):
        print "dataset len: %d / %d" % (self.data_use_len, self.data_len)
        if self.data_use_len == self.data_len:
            return True

        after = int(self.data_use_len * (1 + self.data_growth))
        if after > self.data_len:
            self.data_use_len = self.data_len
        else:
            self.data_use_len = after
        return False # not reach the max length

    def end(self):
        self.is_gen_data = False
        self.batch_loader.join()
        return 

    def initData(self):
        self.data_use_len = self.config['data_init_len']
        self.data_growth = self.config['data_growth']
        self.data_shape = self.config["data_shape"]
        self.batch_len = self.config["batch_len"]
        self.is_gen_data = True
        self.data_lock = threading.Lock()
        self.batch_data = []

        # thread create batches
        self.batch_loader = threading.Thread(target=self.createBatches, args=([1]))
        self.batch_loader.start()
        return

    def createBatches(self, param):
        assert self.batch_len < self.data_use_len

        while self.is_gen_data:
            idx_all = self.data_idx[0:self.data_use_len]

            idx = np.random.permutation(idx_all)[0:self.batch_len]
            X = self.X[idx]
            y = self.y[idx]
            idx = np.random.permutation(idx_all)[0:self.batch_len]
            X_ = self.X[idx]
            y_ = self.y[idx]

            X_final = self.simData(X, X_, y, y_)

            self.data_lock.acquire()
            l = len(self.batch_data)
            self.data_lock.release()
            while l > 0 and self.is_gen_data:
                #print "sleep 1"
                time.sleep(0.001)
                self.data_lock.acquire()
                l = len(self.batch_data)
                self.data_lock.release()

            # do something
            self.data_lock.acquire()
            self.batch_data = [X_final, y]
            self.data_lock.release()

        return

    def loadTests(self):
        Xt = self.pad(self.Xt)
        yt = self.pad(self.yt)

        Xt = self.simData(Xt, Xt, yt, yt)
        return [Xt, yt]

    def loadBaches(self):
        self.data_lock.acquire()
        l = len(self.batch_data)
        self.data_lock.release()
        while l <= 0 and self.is_gen_data:
            #print "sleep 2"
            time.sleep(0.001)
            self.data_lock.acquire()
            l = len(self.batch_data)
            self.data_lock.release()

        data = self.batch_data

        self.data_lock.acquire()
        self.batch_data = []
        self.data_lock.release()
        return data

    def simData(self, X, X_, y, y_):
        d_len = len(X)
        shape_len = len(X.shape)
        #print X.shape
        assert d_len == len(y)

        sim = np.array(map(lambda i: [int(y[i] == y_[i])], range(d_len)))
        s = list(X.shape)
        s[-1] = 1
        sim.shape = tuple(s)

        X_final = np.concatenate((X, X_, sim), shape_len-1)
        return X_final

    def pad(self, data, t=256):
        len_data = len(data)
        n = t - len_data % t
        data = np.concatenate((data, data[0:n]))
        return data

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "data_init_len": 300,
            "data_growth": 0.1,
            "data_shape": (1, 2, 4), #TODO
            "batch_len": 100,
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
    t.end()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

