#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: StockInception.py
# Date: 2016 2016年05月13日 星期五 23时25分16秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.MyCaffeNet import MyCaffeNet

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class StockInception(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: StockInception init")
        self.config = config
        self.debug = config["debug"]
        #super(StockInception, self).__init__(config)
    
    def run(self, fname, param):
        self.netname = fname
        if param.has_key("dropout"):
            self.dropout = param['dropout']
        else:
            self.dropout = 0.5
        #self.inceptionNetTest(2172, 124, 279, param["margin"])
        self.inceptionNet(2172, 124, 279, param["margin"])

    def inceptionNetTest(self, single_dim, stock_dim, num, margin=1):
        net = MyCaffeNet({"debug": True})
        d_dim = 36768
        data, label = net.dataLayer(
                "", "mem", 16,
                tops=["data", "label"],
                memdim=[1,1,d_dim]
                )
        points = [single_dim-stock_dim, single_dim]
        slices = net.sliceLayer(data, points)

        #fc = net.fcLayer(slices[1], 2048, replace='relu')
        #reshape = net.flattenLayer(fc, True)
        #conv01 = net.convLayer(reshape, [80, 1], [16, 1], 128, pad_wh=[0, 0])
        conv01 = net.convLayer(slices[1], [3, 1], [1, 1], 128, pad_wh=[1, 0])

        """
        points = map(lambda x: (x+1) * stock_dim, range(num-1))
        slices_all = net.sliceLayer(slices[2], points)
        tops = []
        one_dim = 64
        for one in slices_all:
            tmp = net.fcLayer(one, one_dim, replace="relu", wname=["d_w", "d_b"])
            tops.append(tmp)
        concat = net.concatLayer(*tops)
        reshape = net.flattenLayer(concat, True)
        conv1 = net.convLayer(reshape, [one_dim, 1], [one_dim, 1], 32)
        pool1 = net.poolLayer(conv1, [34, 1], [2, 1], "ave")
        conv2 = net.convLayer(pool1, [5, 1], [1, 1], 128, pad_wh=[2, 0])

        concat = net.concatLayer(*[conv01, conv2])
        conv_last = net.convLayer(concat, [3, 1], [2, 1], 64, pad_wh=[1, 0])
        reshape = net.flattenLayer(conv1)
        """

        conv1 = net.convLayer(slices[2], [stock_dim, 1], [stock_dim, 1], 64)
        pool1 = net.poolLayer(conv1, [34, 1], [2, 1], "ave")
        concat = net.concatLayer(*[conv01, pool1])
        inception1 = net.inceptionLayerV1(concat)
        pool = net.poolLayer(inception1, [7, 1], [2, 1], "max", pad_wh=[3, 0])
        conv_last = net.convLayer(pool, [3, 1], [1, 1], 64, pad_wh=[1, 0])

        helper1 = net.fcLayer(conv_last, 2048, replace='relu', dropout=0.5)
        helper1_2 = net.fcLayer(helper1, 2048, replace='relu', dropout=0.5)
        helper1_out = net.fcLayer(helper1_2, 20, t="xavier", isout=True)
        helper1_loss = net.lossLayer(helper1_out, label, "softmax", 1)
        helper1_acc = net.accLayer(helper1_out, label)

        classify = net.fcLayer(concat, 20, t="xavier", isout=True)
        loss = net.lossLayer(classify, label, "softmax", 0.3)
        acc = net.accLayer(classify, label)

        ignore1 = net.fcLayer(slices[0], 1, t="xavier", isout=True)
        loss = net.lossLayer(ignore1, label, "softmax", 0)
        #acc = net.accLayer(classify_1, label)

        net.netStr(self.netname)
        return

    def inceptionNet(self, single_dim, stock_dim, num, margin=1):
        net = MyCaffeNet({"debug": True})
        d_dim = 36768
        data, label = net.dataLayer(
                "", "mem", 16,
                tops=["data", "label"],
                memdim=[1,1,d_dim]
                )
        points = [single_dim-stock_dim, single_dim]
        slices = net.sliceLayer(data, points)

        conv01 = net.convLayer(slices[1], [3, 1], [1, 1], 128, pad_wh=[1, 0])

        conv1 = net.convLayer(slices[2], [stock_dim, 1], [stock_dim, 1], 64)
        pool1 = net.poolLayer(conv1, [34, 1], [2, 1], "ave")

        concat = net.concatLayer(*[conv01, pool1])
        inception1 = net.inceptionLayerV1(concat)
        pool = net.poolLayer(inception1, [7, 1], [2, 1], "max", pad_wh=[3, 0])
        conv_last = net.convLayer(pool, [3, 1], [1, 1], 64, pad_wh=[1, 0])

        helper1 = net.fcLayer(conv_last, 2048, replace='relu', dropout=self.dropout)
        helper1_2 = net.fcLayer(helper1, 2048, replace='relu', dropout=self.dropout)
        helper1_out = net.fcLayer(helper1_2, 20, t="xavier", isout=True)
        helper1_loss = net.lossLayer(helper1_out, label, "softmax", 1)
        helper1_acc = net.accLayer(helper1_out, label)

        classify = net.fcLayer(concat, 20, t="xavier", isout=True)
        loss = net.lossLayer(classify, label, "softmax", 0.2)
        acc = net.accLayer(classify, label)

        ignore1 = net.fcLayer(slices[0], 1, t="xavier", isout=True)
        loss = net.lossLayer(ignore1, label, "softmax", 0)

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
    t = StockInception(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

