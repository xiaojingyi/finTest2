#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: RunerTest.py
# Date: 2016 2016年05月08日 星期日 18时44分43秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import sklearn
import sklearn.datasets
import sklearn.linear_model
import numpy as np
sys.path.append("/datas/lib/py")
from App import App
from NetFactory import NetFactory

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class RunerTest(App):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: RunerTest init")
        self.config = config
        self.debug = config["debug"]
        super(RunerTest, self).__init__(config)
    
    def run(self):
        queue=["maxdata", "convergency", "tearing", "ignore", "rst", "tearing"]
        if self.states["margin"] <= 1:
            del queue[0]
        self.runBase(queue)

    def mkData(self):
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
            l = len(y)
            X.shape = (l, 1, 1, 3000)
            y.shape = (l, 1, 1, 1)
            l = len(yt)
            Xt.shape = (l, 1, 1, 3000)
            yt.shape = (l, 1, 1, 1)
            np.save(fname_X, X)
            np.save(fname_y, y)
            np.save(fname_Xt, Xt)
            np.save(fname_yt, yt)
        print "data prepared."
        return X, y, Xt, yt

    def mkNet(self, margin=1):
        f = NetFactory(self.config)
        f.init("Net", {"margin": margin})
        return

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            # training param
            "gpu_id": 1,
            "lr": 0.0001,
            "gamma": 0.5,
            "stop_lr": 0.000001,
            "margin": 1,
            "margin_max": 4,
            "model": sys.argv[1],
            "solver_template": "tpls/solver_adadelta.prototxt.template",
            "continue_sameloss_breaker": 20, # n * threshold_check steps

            # training thresholds
            "threshold_check": 1000, # the steps to do checking
            "threshold_acc": 0.98, # train set accuracy
            "threshold_loss": 0.02, # train set loss

            # dataset config
            "data_init_len": 500,
            "cluster_number": 2,
            "data_growth": 0.2,
            "data_shape": (1, 2, 4), #TODO
            "batch_len": 32 * 10,
            "data_type": "DataMem",
            }
    t = RunerTest(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

