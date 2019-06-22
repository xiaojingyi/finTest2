#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: SauronDecoderNet.py
# Date: 2016 2016年02月29日 星期一 14时05分48秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
sys.path.append("lib")
from NetBuilder import NetBuilder
from lib.MyCaffe import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class SauronDecoderNet(NetBuilder):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: SauronDecoderNet init")
        self.config = config
        self.debug = config["debug"]
        super(SauronDecoderNet, self).__init__(config)
        self.losses = []
    
    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

    def netv2(self, dim_n, net_name="RecNetV2.prototxt"):
        data, label = dataLayer("./abs.h5", "lmdb")
        fc1, top = fcLayer(data, 100, "sigmoid")
        """
        fc2, top = fcLayer(top, 10, "sigmoid")
        fc2, top = fcLayer(top, 2048, "relu")
        """
        fc3, top = fcLayer(top, dim_n, "sigmoid")
        self.losses.append(lossLayer(data, top, "eloss"))

        loss1 = lossLayer(label, label, "eloss", 0)
        self.losses.append(loss1)
        saveNet(net_name, self.losses, "SauronDecoderNet V2")
        return

    def netTest(self, dim_n, net_name="test.prototxt"):
        step = self.config['step_num']
        data, label = dataLayer("./abs.h5", "lmdb")
        split_dparam = {"data": data, "dim": dim_n, "loss": 1}
        split_1, dim_num = self.splitDatas(data, dim_n, step, split_dparam)
        split_dparam["loss"] = 0.3
        split_2, dim_num = self.splitDatas(split_1, dim_num, step, split_dparam)
        #data2, dim_num = self.joinData(split_1, dim_num, dim_n)
        print dim_n, dim_num
        #loss = lossLayer(data2, data, "eloss", 1.0)
        #self.losses.append(loss)
        loss1 = lossLayer(label, label, "eloss", 0)
        self.losses.append(loss1)
        saveNet(net_name, self.losses, "SauronDecoderNet V1")
        return

def main():
    conf = {
            "debug": True,
            "step_num": int(sys.argv[2]),
            }
    t = SauronDecoderNet(conf)
    t.net(int(sys.argv[1]), "train.prototxt")
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

