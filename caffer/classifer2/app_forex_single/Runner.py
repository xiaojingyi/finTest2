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
            "gpu_id": sys.argv[2],
            "max_iter": 10000000,
            "lr": 0.00001,
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

