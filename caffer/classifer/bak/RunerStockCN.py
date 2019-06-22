#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: RunerStockCN.py
# Date: 2016 2016年05月11日 星期三 23时00分10秒
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

class RunerStockCN(App):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: RunerStockCN init")
        self.config = config
        self.debug = config["debug"]
        super(RunerStockCN, self).__init__(config)
    
    def mkNet(self, margin=1, dropout=0.5):
        f = NetFactory(self.config)
        netname = f.init("StockAlex", {"margin": margin, "dropout": dropout}, "gpu%d_" % self.gpu_id + "alex.prototxt")
        return netname

    def run(self):
        queue=["maxdata", "convergency", "tearing", ]
        #if self.states["margin"] <= 1:
        #    del queue[0]
        self.runBase(queue)

    def mkData(self):
        X, y, y_ = self.loadH5("/.jingyi/codes/stockcn2/caffe/data_0.h5")
        Xt, yt, yt_ = self.loadH5("/.jingyi/codes/stockcn2/caffe/data_1.h5")
        return X, y, Xt, yt, yt_

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
            "gamma": 0.95,
            "stop_lr": 0.00001,
            "margin": 1,
            "margin_max": 127,
            "model": sys.argv[1],
            #"solver_template": "tpls/solver_adadelta.prototxt.template",
            "solver_template": "tpls/solver_sgd.prototxt.template",
            "continue_sameloss_breaker": 17, # n * threshold_check steps

            # training thresholds
            "threshold_check": 1000, # the steps to do checking
            "threshold_acc": 0.98, # train set accuracy
            "threshold_loss": 0.008, # train set loss

            # dataset config
            "data_init_len": 300,
            "cluster_number": 3,
            "data_growth": 0.2,
            "data_shape": (1, 1, 4012), #TODO
            "batch_len": 256 * 1,
            "dcache_prefix": "alex",
            "data_type": "DataMem",
            "is_sim": False,
            }
    t = RunerStockCN(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

