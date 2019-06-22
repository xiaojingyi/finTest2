#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Runner.py
# Date: 2016 2016年06月04日 星期六 22时10分54秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../libs")
from App import App
from NetFactory import NetFactory

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Runner(App):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Runner init")
        self.config = config
        self.debug = config["debug"]
        super(Runner, self).__init__(config)
    
    def mkNet(self, margin=1, dropout=0.5):
        f = NetFactory(self.config)
        netname = f.init("TestNet", {"margin": margin, "dropout": dropout}, "gpu%d_" % self.gpu_id + "test.prototxt")
        return netname

    def run(self):
        queue=[
                "maxdata",
                "convergency",
                ]
        self.runBase(queue)

    def mkData(self):
        dim = 1000
        X, y = sklearn.datasets.make_classification(
            n_samples=5000, n_features=dim, n_redundant=0, n_informative=32, 
            n_classes=2,
            n_clusters_per_class=2, hypercube=False, random_state=0
            )
        #X = np.random.random((5000, dim))
        X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y, test_size=0.2)
        l = len(y)
        X.shape = (l, 1, 1, dim)
        y.shape = (l, 1, 1, 1)
        l = len(yt)
        Xt.shape = (l, 1, 1, dim)
        yt.shape = (l, 1, 1, 1)

        return X, y, Xt, yt, []

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            # training param
            "gpu_id": sys.argv[2], 
            "max_iter": 5000,
            "continue_max_iter": 20000,
            "lr": 0.001,
            "gamma": 0.99,
            "stop_lr": 0.00001,
            "margin": 1,
            "margin_max": 127,
            "model": sys.argv[1],
            "solver_template": "./solver_sgd.prototxt.template",
            "continue_sameloss_breaker": 17, # n * threshold_check steps

            # training thresholds
            "threshold_check": 1000, # the steps to do checking
            "threshold_acc": 0.98, # train set accuracy
            "threshold_loss": 0.000008, # train set loss

            # dataset config
            "data_init_len": 500,
            "cluster_number": 3,
            "data_growth": 0.2,
            "data_shape": (1, 1, 4012), #TODO
            "batch_len": 200 * 1,
            "dcache_prefix": "alex",
            "data_type": "DataTest",
            "is_sim": False,
            }
    t = Runner(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

