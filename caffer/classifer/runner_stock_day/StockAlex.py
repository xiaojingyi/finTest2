#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: StockAlex.py
# Date: 2016 2016年05月11日 星期三 23时05分47秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.MyCaffeNet import MyCaffeNet

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class StockAlex(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: StockAlex init")
        self.config = config
        self.debug = config["debug"]
        #super(StockAlex, self).__init__(config)
    
    def run(self, fname, param):
        self.netname = fname
        if param.has_key("dropout"):
            self.dropout = param['dropout']
        else:
            self.dropout = 0.7
        #self.alexNet2(2172, 124, 279, param["margin"])
        self.testNet(2172, 124, 279, param["margin"])
        #self.testNetSim(2172, 124, 279, param["margin"])
        #self.alexNet(2172, 124, 279, param["margin"])

    def testNet(self, single_dim, stock_dim, num, margin=1):
        net = MyCaffeNet({"debug": True})
        d_dim = 4012
        data, label = net.dataLayer(
                "", "mem", 128,
                tops=["data", "label"],
                memdim=[1,1,d_dim]
                )
        """
        """
        points = [12]
        slices = net.sliceLayer(data, points)

        fc = net.fcLayer(slices[0], 1024, replace="relu")

        #drop = slices[1]
        drop = net.groupDropLayer(slices[1], 800, 400, False)
        conv1 = net.convLayer(drop, [5, 1], [5, 1], 64)
        layer = net.normLayer(conv1)
        pool1 = net.poolLayer(layer, [6, 1], [2, 1], "max", pad_wh=[2, 0])
        conv2 = net.convLayer(pool1, [4, 1], [2, 1], 32, pad_wh=[1, 0])
        #conv_tmp = net.convLayer(layer, [6, 1], [2, 1], 64, pad_wh=[2, 0])
        #conv2 = net.convLayer(conv_tmp, [4, 1], [2, 1], 32, pad_wh=[1, 0])
        conv3 = net.convLayer(conv2, [4, 1], [2, 1], 16, pad_wh=[1, 0])
        flat = net.flattenLayer(conv3)

        concat = net.concatLayer(*[fc, flat])

        helper1 = net.fcLayer(concat, 2048, t="xavier", 
                replace='relu', dropout=self.dropout)
        helper1_2 = net.fcLayer(helper1, 2048, t="xavier",
                replace='relu', dropout=self.dropout)

        helper1_out = net.fcLayer(helper1_2, 20, t="xavier", isout=True)
        helper1_loss = net.lossLayer(helper1_out, label, "softmax", 1)
        helper1_acc = net.accLayer(helper1_out, label)

        net.netStr(self.netname)
        return

    def testNetSim(self, single_dim, stock_dim, num, margin=1):
        net = MyCaffeNet({"debug": True})
        d_dim = 4012
        data, label = net.dataLayer(
                "", "mem", 64,
                tops=["data", "label"],
                memdim=[1,1,d_dim*2+1]
                )
        datas = net.sliceLayer(data, [d_dim, d_dim*2])
        sim = datas[2]
        points = [12]

        " the normal side"
        slices = net.sliceLayer(datas[0], points)
        fc = net.fcLayer(slices[0], 1024, replace="relu", wname=["fc1_w", "fc1_b"], lr=[1, 2])

        drop = net.groupDropLayer(slices[1], 800, 600, False)
        conv1 = net.convLayer(drop, [5, 1], [5, 1], 64, wname=["conv1_w", "conv1_b"], lr=[1, 2])
        layer = net.normLayer(conv1)
        pool1 = net.poolLayer(layer, [6, 1], [2, 1], "ave", pad_wh=[2, 0])
        conv2 = net.convLayer(pool1, [4, 1], [2, 1], 32, pad_wh=[1, 0], wname=["conv2_w", "conv2_b"], lr=[1, 2])
        conv3 = net.convLayer(conv2, [4, 1], [2, 1], 16, pad_wh=[1, 0], wname=["conv3_w", "conv3_b"], lr=[1, 2])
        flat = net.flattenLayer(conv3)

        concat = net.concatLayer(*[fc, flat])

        fc = net.fcLayer(concat, 2048, replace='relu', dropout=0, wname=["fc2_w", "fc2_b"], lr=[1, 2])
        drop = net.dropLayer(fc, self.dropout)
        fc = net.fcLayer(drop, 2048, replace='relu', dropout=0, wname=["fc3_w", "fc3_b"], lr=[1, 2])
        drop1 = net.dropLayer(fc, self.dropout)
        decoder = net.fcLayer(drop1, 128, replace="sigmoid", wname=["dw", "db"], lr=[1, 2])
        " the normal side start"

        out = net.fcLayer(drop1, 20, t="xavier", isout=True)

        " the pair side start"
        slices = net.sliceLayer(datas[1], points)
        fc = net.fcLayer(slices[0], 1024, replace="relu", wname=["fc1_w", "fc1_b"], lr=[1, 2])

        drop = net.groupDropLayer(slices[1], 800, 600, False)
        conv1 = net.convLayer(drop, [5, 1], [5, 1], 64, wname=["conv1_w", "conv1_b"], lr=[1, 2])
        layer = net.normLayer(conv1)
        pool1 = net.poolLayer(layer, [6, 1], [2, 1], "ave", pad_wh=[2, 0])
        conv2 = net.convLayer(pool1, [4, 1], [2, 1], 32, pad_wh=[1, 0], wname=["conv2_w", "conv2_b"], lr=[1, 2])
        conv3 = net.convLayer(conv2, [4, 1], [2, 1], 16, pad_wh=[1, 0], wname=["conv3_w", "conv3_b"], lr=[1, 2])
        flat = net.flattenLayer(conv3)

        concat = net.concatLayer(*[fc, flat])

        fc = net.fcLayer(concat, 2048, replace='relu', dropout=0, wname=["fc2_w", "fc2_b"], lr=[1, 2])
        drop = net.dropLayer(fc, self.dropout)
        fc = net.fcLayer(drop, 2048, replace='relu', dropout=0, wname=["fc3_w", "fc3_b"], lr=[1, 2])
        drop1_ = net.dropLayer(fc, self.dropout)
        decoder_ = net.fcLayer(drop1_, 128, replace="sigmoid", wname=["dw", "db"], lr=[1, 2])
        " the pair side end"

        loss = net.lossLayer(out, label, "softmax", 0.01)
        acc = net.accLayer(out, label)

        loss_sim = net.lossLayer(decoder, decoder_, "contrastive", 1, third_bottom=sim, param={"margin": margin})
        net.netStr(self.netname)
        return

    def alexNet2(self, single_dim, stock_dim, num, margin=1):
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
        conv2 = net.convLayer(concat, [3, 1], [1, 1], 128, pad_wh=[1, 0])
        pool = net.poolLayer(conv2, [7, 1], [2, 1], "max", pad_wh=[3, 0])
        conv_last = net.convLayer(pool, [3, 1], [1, 1], 64, pad_wh=[1, 0])

        helper1 = net.fcLayer(conv_last, 2048, replace='relu', dropout=self.dropout)
        helper1_2 = net.fcLayer(helper1, 2048, replace='relu', dropout=self.dropout)
        helper1_out = net.fcLayer(helper1_2, 20, t="xavier", isout=True)
        helper1_loss = net.lossLayer(helper1_out, label, "softmax", 1)
        helper1_acc = net.accLayer(helper1_out, label)

        ignore1 = net.fcLayer(slices[0], 1, t="xavier", isout=True)
        loss = net.lossLayer(ignore1, label, "softmax", 0)

        net.netStr(self.netname)
        return

    def alexNet(self, single_dim, stock_dim, num, margin=1):
        net = MyCaffeNet({"debug": True})
        d_dim = 36768
        data, label = net.dataLayer(
                "", "mem", 32,
                tops=["data", "label"],
                memdim=[1,1,d_dim*2+1]
                )
        datas = net.sliceLayer(data, [d_dim, d_dim*2])
        sim = datas[2]
        points = [single_dim]
        dim_nn = 2048
        drop_out = 0.7

        " the normal side"
        slices = net.sliceLayer(datas[0], points)

        fc = net.fcLayer(slices[0], dim_nn, replace="relu", wname=["fc1_w", "fc1_b"], lr=[1, 2])
        fc = net.ippLayer(fc, dim_nn, wname=["ipp_w", "ipp_b"], lr=[1, 2])

        conv1 = net.convLayer( slices[1], [stock_dim, 1], [stock_dim, 1], 64, wname=["conv1_w", "conv1_b"], lr=[1, 2])
        pool1 = net.poolLayer(conv1, [5, 1], [1, 1], "ave")
        norm1 = net.normLayer(pool1)
        conv2 = net.convLayer(norm1, [5, 1], [2, 1], 64, wname=["conv2_w", "conv2_b"], lr=[1, 2])
        pool2 = net.poolLayer(conv2, [5, 1], [1, 1], "ave")
        conv3 = net.convLayer(pool2, [5, 1], [1, 1], 32, pad_wh=[4,0], wname=["conv3_w", "conv3_b"], lr=[1, 2])
        pool3 = net.poolLayer(conv3, [5, 1], [2, 1], "max")

        flat1 = net.flattenLayer(pool3)
        concat = net.concatLayer(*[fc, flat1])

        top1 = net.fcLayer(concat, dim_nn, replace="relu", dropout=drop_out, wname=["fc2_w", "fc2_b"], lr=[1, 2])
        top2 = net.fcLayer(top1, dim_nn, replace="relu", dropout=drop_out, wname=["fc3_w", "fc3_b"], lr=[1, 2])
        decoder = net.fcLayer(top2, 128, replace="sigmoid", wname=["dw", "db"], lr=[1, 2])
        " the normal side start"

        " the pair side start"
        slices = net.sliceLayer(datas[1], points)

        fc = net.fcLayer(slices[0], dim_nn, replace="relu", wname=["fc1_w", "fc1_b"], lr=[1, 2])
        fc = net.ippLayer(fc, dim_nn, wname=["ipp_w", "ipp_b"], lr=[1, 2])

        conv1 = net.convLayer( slices[1], [stock_dim, 1], [stock_dim, 1], 64, wname=["conv1_w", "conv1_b"], lr=[1, 2])
        pool1 = net.poolLayer(conv1, [5, 1], [1, 1], "ave")
        norm1 = net.normLayer(pool1)
        conv2 = net.convLayer(norm1, [5, 1], [2, 1], 64, wname=["conv2_w", "conv2_b"], lr=[1, 2])
        pool2 = net.poolLayer(conv2, [5, 1], [1, 1], "ave")
        conv3 = net.convLayer(pool2, [5, 1], [1, 1], 32, pad_wh=[4,0], wname=["conv3_w", "conv3_b"], lr=[1, 2])
        pool3 = net.poolLayer(conv3, [5, 1], [2, 1], "max")

        flat1 = net.flattenLayer(pool3)
        concat = net.concatLayer(*[fc, flat1])

        top1 = net.fcLayer(concat, dim_nn, replace="relu", dropout=drop_out, wname=["fc2_w", "fc2_b"], lr=[1, 2])
        top2_ = net.fcLayer(top1, dim_nn, replace="relu", dropout=drop_out, wname=["fc3_w", "fc3_b"], lr=[1, 2])
        decoder_ = net.fcLayer(top2_, 128, replace="sigmoid", wname=["dw", "db"], lr=[1, 2])
        " the pair side end"

        top3 = net.fcLayer(top2, 20, isout=True, lr=[1, 2])
        loss = net.lossLayer(top3, label, "softmax", 1)
        loss_sim = net.lossLayer(decoder, decoder_, "contrastive", 0, third_bottom=sim, param={"margin": margin})
        acc = net.accLayer(top3, label)
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
    t = StockAlex(conf)
    t.run("", {"margin": 1})
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

