#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: RunerStockInception.py
# Date: 2016 2016年05月13日 星期五 23时39分59秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import h5py
import numpy as np
sys.path.append("/datas/lib/py")
from App import App
from NetFactory import NetFactory

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class RunerStockInception(App):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: RunerStockInception init")
        self.config = config
        self.debug = config["debug"]
        super(RunerStockInception, self).__init__(config)
    
    def mkNet(self, margin=1, dropout=0.5):
        f = NetFactory(self.config)
        netname = f.init("StockInception", {"margin": margin, "dropout": dropout}, "gpu%d_" % self.gpu_id + "inception.prototxt")
        return netname

    def run(self):
        queue=[
                "maxdata", 
                "convergency", 
                "dropmax",
                ]
        self.runBase(queue)

    def mkData(self):
        X, y, y_ = self.loadH5("/.jingyi/codes/stockcn/caffe/data_0.h5")
        Xt, yt, yt_ = self.loadH5("/.jingyi/codes/stockcn/caffe/data_1.h5")
        return X, y, Xt, yt

    def loadH5(self, fname):
        with h5py.File(fname, 'r') as f:
            X = f['data'][:]
            y = f['label'][:]
            y_ = f['realy'][:]

        return np.array(X).astype(np.float32), \
                np.array(y).astype(np.float32).reshape(len(y), 1, 1, 1), \
                np.array(y_).astype(np.float32)

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
            "lr": 0.001,
            "gamma": 0.99,
            "stop_lr": 0.000001,
            "margin": 1,
            "margin_max": 127,
            "model": sys.argv[1],
            #"solver_template": "tpls/solver_adadelta.prototxt.template",
            "solver_template": "tpls/solver_sgd.prototxt.template",
            "continue_sameloss_breaker": 21, # n * threshold_check steps

            # training thresholds
            "threshold_check": 1000, # the steps to do checking
            "threshold_acc": 0.98, # train set accuracy
            "threshold_loss": 0.008, # train set loss

            # dataset config
            "data_init_len": 10000,
            "cluster_number": 3,
            "data_growth": 0.1,
            "data_shape": (1, 2, 4), #TODO
            "batch_len": 64 * 5,
            "data_type": "DataMem",
            "is_sim": False,
            }
    t = RunerStockInception(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

