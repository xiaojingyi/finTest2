#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: trainer.py
# Date: 2016 2016年04月22日 星期五 17时54分17秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import random
import sklearn
import sklearn.datasets
import sklearn.linear_model
sys.path.append("/datas/lib/py")
sys.path.append("/.jingyi/codes/caffer/lib")
from CaffeTrainerPython import CaffeTrainerPython

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class trainer(CaffeTrainerPython):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: trainer init")
        self.config = config
        self.debug = config["debug"]
        super(trainer, self).__init__(config)
        self.initData()

    def initData(self):
        fname_X = "X.npy"
        fname_y = "y.npy"
        fname_Xt = "Xt.npy"
        fname_yt = "yt.npy"
        if os.path.exists(fname_X):
            X = np.load(fname_X)
            y = np.load(fname_y)
            Xt = np.load(fname_Xt)
            yt = np.load(fname_yt)
        else:
            X, y = sklearn.datasets.make_classification(
                n_samples=5000, n_features=3000, n_redundant=500, n_informative=2000, 
                n_classes=2,
                n_clusters_per_class=3, hypercube=False, random_state=0
                )
            X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y, test_size=0.2)
            np.save(fname_X, X)
            np.save(fname_y, y)
            np.save(fname_Xt, Xt)
            np.save(fname_yt, yt)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        Xt = Xt.astype(np.float32)
        yt = yt.astype(np.float32)
        self.Xy = (X, y)
        self.Xyt = (Xt, yt)
        self.Xlen = len(y)
        print "data prepared."
        return

    def dataLoader(self, nlen):
        idx = np.random.permutation(np.arange(self.Xlen))[0:nlen]
        Xy = (self.Xy[0][idx], self.Xy[1][idx])
        idx = np.random.permutation(np.arange(self.Xlen))[0:nlen]
        Xy_ = (self.Xy[0][idx], self.Xy[1][idx])
        sim = np.array(map(lambda i: [int(Xy[1][i] == Xy_[1][i])], range(len(Xy[1]))))
        resXy = (np.hstack((Xy[0], Xy_[0], sim)).astype(np.float32), Xy[1])
        resXy[0].shape = (nlen, 1, 1, 6001)
        resXy[1].shape = (nlen, 1, 1, 1)
        #print resXy[0]
        #print resXy[1]

        Xyt = self.Xyt
        lenXyt = len(Xyt[1])
        sim = np.array(map(lambda i: [1], range(lenXyt)))
        resXyt = (np.hstack((Xyt[0], Xyt[0], sim)).astype(np.float32), Xyt[1])
        resXyt[0].shape = (lenXyt, 1, 1, 6001)
        resXyt[1].shape = (lenXyt, 1, 1, 1)
        return resXy, resXyt, nlen
    
    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "solver": sys.argv[1],
            "weights": sys.argv[2],
            "gpu_id": int(sys.argv[3]),
            "nbatch": 10*32,
            }
    t = trainer(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

