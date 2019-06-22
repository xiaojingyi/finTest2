#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: StockRangeWeek.py
# Date: 2016 2016年05月29日 星期日 19时58分36秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.MyCaffeNet import MyCaffeNet

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class StockRangeWeek(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: StockRangeWeek init")
        self.config = config
        self.debug = config["debug"]
        #super(StockRangeWeek, self).__init__(config)
    
    def run(self, fname, param):
        self.netname = fname
        if param.has_key("dropout"):
            self.dropout = param['dropout']
        else:
            self.dropout = 0.7
        self.alexNet(param["margin"])

    def alexNet(self, margin=1):
        net = MyCaffeNet({"debug": True})
        onelen = 24
        d_dim = onelen + onelen * 800
        data, label = net.dataLayer(
                "", "mem", 32,
                tops=["data", "label"],
                memdim=[1,1,d_dim]
                )
        points = [onelen]
        slices = net.sliceLayer(data, points)

        fc = net.fcLayer(slices[0], 1024, replace="relu")

        conv1 = net.convLayer(slices[1], [onelen, 1], [onelen, 1], 64)
        layer = net.normLayer(conv1)
        pool1 = net.poolLayer(layer, [6, 1], [2, 1], "ave", pad_wh=[2, 0])
        conv2 = net.convLayer(pool1, [4, 1], [2, 1], 32, pad_wh=[1, 0])
        conv3 = net.convLayer(conv2, [4, 1], [2, 1], 16, pad_wh=[1, 0])
        flat = net.flattenLayer(conv3)

        concat = net.concatLayer(*[fc, flat])

        helper1 = net.fcLayer(concat, 2048, replace='relu', dropout=self.dropout)
        helper1_2 = net.fcLayer(helper1, 2048, replace='relu', dropout=self.dropout)
        helper1_out = net.fcLayer(helper1_2, 20, t="xavier", isout=True)
        helper1_loss = net.lossLayer(helper1_out, label, "softmax", 1)
        helper1_acc = net.accLayer(helper1_out, label)

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
    t = StockRangeWeek(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

