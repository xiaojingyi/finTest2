#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Runner.py
# Date: 2016 Mon 01 Aug 2016 07:05:38 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../libs")
from lib.MyMath import *
from App import App
from Net import Net

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Runner(App):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Runner init")
        self.config = config
        self.debug = config["debug"]
        super(Runner, self).__init__(config)
        self.out_index = 1
    
    def mkNet(self, margin=1, dropout=0.5):
        net = Net({"debug": True, "dropout": dropout})
        netname = "gpu%d_" % self.gpu_id + "stock.train.prototxt"
        netname_t = "gpu%d_" % self.gpu_id + "stock.test.prototxt"
        net.trainNet(netname)
        net.testNet(netname_t)
        return netname, netname_t 

    def testCallback(self, blobs, bsize, i=0, i_max=0):
        print "in callback"
        res = []
        for j in range(bsize):
            X_ = blobs["pridata"].data[j]
            X = blobs["data"].data[j]
            shape = X.shape[-1]
            X.shape = shape
            X_.shape = shape
            up_min = 1
            down_max = 0
            for k in range(shape):
                if X_[k] > 0:
                    if up_min > X[k]:
                        up_min = X[k]
                elif X_[k] < 0:
                    if down_max < X[k]:
                        down_max = X[k]

            one = blobs["sigmoid_1"].data[j]
            one.shape = shape
            for k in range(shape):
                label = 1 if X[k] >= up_min else (
                        2 if X[k] <= down_max else 0)
                label_ = 1 if one[k] >= up_min else (
                        2 if one[k] <= down_max else 0)
                res.append([label_, label, X_[k]])
        print blobs["loss_1"].data
        print blobs["loss_2"].data
        print np.array(res)
        return res

    def run(self):
        queue=["convergency",]
        self.runBase(queue)

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            # training param
            "out_blob": "sigmoid_1",
            "test_len": 1000,
            "gpu_id": sys.argv[2],
            "max_iter": 10000000,
            "lr": 0.0001,
            "gamma": 1.0,
            "dropout": 0.5,
            "stop_lr": 0.000001,
            "margin": 1,
            "margin_max": 127,
            "model": sys.argv[1],
            #"solver_template": "solver_rmsprop.prototxt.template",
            #"solver_template": "solver_adagrad.prototxt.template",
            "solver_template": "solver_adam.prototxt.template",
            "continue_sameloss_breaker": 17,

            # training thresholds
            "threshold_check": 100, # the steps to do checking
            "threshold_acc": 0.98, # train set accuracy
            "threshold_loss": 0.0001, # train set loss

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

