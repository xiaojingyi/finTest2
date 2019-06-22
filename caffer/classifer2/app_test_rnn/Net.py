#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: NetTest.py
# Date: 2016 Mon 01 Aug 2016 07:18:45 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.MyCaffeNet import MyCaffeNet

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class NetTest(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: NetTest init")
        self.config = config
        self.debug = config["debug"]
        #super(NetTest, self).__init__(config)
        self.dropout = 0.6
    
    def trainNet(self, fname):
        self.fcNet(fname, False)

    def testNet(self, fname):
        self.fcNet(fname, True)

    def fcNet(self, netname, is_test):
        net = MyCaffeNet({"debug": True})
        data, label = net.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                )
        fc = net.fcLayer(data, 512, replace="relu")
        fc = net.fcLayer(fc, 512, replace="relu")
        fc = net.fcLayer(fc, 512, replace="relu")

        out = net.fcLayer(fc, 20, t="xavier", isout=True)
        loss = net.lossLayer(out, label, "softmax", 1)
        acc = net.accLayer(out, label)

        net.netStr(netname)
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
    t = NetTest(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

