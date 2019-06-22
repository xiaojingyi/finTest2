#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: TestNet.py
# Date: 2016 2016年06月04日 星期六 22时22分21秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.MyCaffeNet import MyCaffeNet

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class TestNet(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: TestNet init")
        self.config = config
        self.debug = config["debug"]
        #super(TestNet, self).__init__(config)
    
    def run(self, fname, param):
        self.netname = fname
        if param.has_key("dropout"):
            self.dropout = param['dropout']
        else:
            self.dropout = 0.7
        self.testNet(param["margin"])
        #self.fcNet(param["margin"])

    def fcNet(self, margin=1):
        net = MyCaffeNet({"debug": True})
        d_dim = 1000
        data, label = net.dataLayer(
                "", "mem", 20,
                memdim=[1,1,d_dim+1]
                )
        fc = net.fcLayer(data, 512, replace="relu")
        fc = net.fcLayer(fc, 512, replace="relu")
        fc = net.fcLayer(fc, 512, replace="relu")

        out = net.fcLayer(fc, 20, t="xavier", isout=True)
        loss = net.lossLayer(out, label, "softmax", 1)
        acc = net.accLayer(out, label)

        net.netStr(self.netname)
        return

    def testNet(self, margin=1):
        net = MyCaffeNet({"debug": True})
        d_dim = 1000
        data, label = net.dataLayer(
                "", "mem", 20,
                memdim=[1,1,d_dim+1]
                )
        data, clip = net.sliceLayer(data, [d_dim])
        fc = net.rnnLayer(data, clip, 512, is_lstm=True)
        fc = net.fcLayer(fc, 512, replace="relu")
        fc = net.fcLayer(fc, 512, replace="relu")

        out = net.fcLayer(fc, 20, t="xavier", isout=True)
        loss = net.lossLayer(out, label, "softmax", 1)
        acc = net.accLayer(out, label)

        net.netStr(self.netname)
        return

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = TestNet(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

