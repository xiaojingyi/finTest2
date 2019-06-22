#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Net.py
# Date: 2016 2016年07月03日 星期日 19时10分17秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.MyCaffeNet import MyCaffeNet

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Net(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Net init")
        self.config = config
        self.debug = config["debug"]
        self.dropout = config['dropout']
        #super(Net, self).__init__(config)
    
    def trainNet(self, fname):
        self.net(fname, False)
    
    def testNet(self, fname):
        self.net(fname, True)

    def net(self, netname, is_test):
        net = MyCaffeNet({"debug": True})
        data, label_muti, label, label_ = net.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                tops = ["data", "label_muti", "label", "label_"]
                )

        points = [12]
        slices = net.sliceLayer(data, points)
        net.silenceLayer(label_)

        fc = net.fcLayer(slices[0], 1024, replace="relu")

        drop = net.groupDropLayer(slices[1], 800, 400, False)
        conv1 = net.convLayer(drop, [5, 1], [5, 1], 64)
        layer = net.normLayer(conv1)
        #pool1 = net.poolLayer(layer, [6, 1], [2, 1], 
        #        "ave", pad_wh=[2, 0])
        pool1 = net.convLayer(layer, [6, 1], [2, 1], 
                64, pad_wh=[2, 0])
        conv2 = net.convLayer(pool1, [4, 1], [2, 1], 
                32, pad_wh=[1, 0])
        conv3 = net.convLayer(conv2, [4, 1], [2, 1], 
                16, pad_wh=[1, 0])
        flat = net.flattenLayer(conv3)

        concat = net.concatLayer(*[fc, flat])

        fc = net.fcLayer(concat, 2048, t="xavier", 
                replace='relu', dropout=self.dropout)
        fc = net.fcLayer(fc, 2048, t="xavier",
                replace='relu', dropout=self.dropout)

        out = net.fcLayer(fc, 20, t="xavier", isout=True)

        loss = net.lossLayer(out, label_muti, "sigcross", 1)
        #net.silenceLayer(label_muti)
        #loss = net.lossLayer(out, label, "softmax", 1)
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
    t = Net(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

