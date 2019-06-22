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
        self.net_model = MyCaffeNet({"debug": True})
        self.net(fname, False)
    
    def testNet(self, fname):
        self.net_model = MyCaffeNet({"debug": True})
        self.net(fname, True)

    def atomLayers(self, bottom, label, label_muti, onelen, loss_num, is_test, ones, zeros):
        points = [7 + onelen]
        slices = self.net_model.sliceLayer(bottom, points)

        fc = self.net_model.fcLayer(slices[0], 1024, replace="relu")

        drop = self.net_model.groupDropLayer(slices[1], 800, 400, False)
        conv1 = self.net_model.convLayer(drop, [onelen, 1], [onelen, 1], 64)
        layer = self.net_model.normLayer(conv1)
        pool1 = self.net_model.convLayer(layer, [6, 1], [2, 1], 
                64, pad_wh=[2, 0])
        conv2 = self.net_model.convLayer(pool1, [4, 1], [2, 1], 
                32, pad_wh=[1, 0])
        conv3 = self.net_model.convLayer(conv2, [4, 1], [2, 1], 
                16, pad_wh=[1, 0])
        flat = self.net_model.flattenLayer(conv3)

        concat = self.net_model.concatLayer(*[fc, flat])

        n_num = 2048
        fc = self.net_model.fcLayer(concat, n_num, t="xavier", 
                replace='relu')
        fc_drop = self.net_model.sameDropLayer(ones, fc, False)
        fc_drop = self.net_model.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net_model.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net_model.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net_model.sameDropLayer(fc_drop, fc, False)
        loss = self.net_model.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net_model.sameDropLayer(fc, fc_drop, False)
        fc = self.net_model.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net_model.scaleLayer(fc, in_place=False)

        fc = self.net_model.fcLayer(fc, n_num, t="xavier",
                replace='relu')
        fc_drop = self.net_model.sameDropLayer(ones, fc, False)
        fc_drop = self.net_model.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net_model.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net_model.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net_model.sameDropLayer(fc_drop, fc, False)
        loss = self.net_model.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net_model.sameDropLayer(fc, fc_drop, False)
        fc = self.net_model.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net_model.scaleLayer(fc, in_place=False)

        out = self.net_model.fcLayer(fc, 20, t="xavier", isout=True)
        out_seg = self.net_model.sigmoidLayer(out)
        self.net_model.silenceLayer(out_seg)
        loss = self.net_model.lossLayer(out, label_muti, "sigcross", loss_num)
        acc = self.net_model.accLayer(out, label)
        return fc

    def net(self, netname, is_test):
        data, label_muti, label, label_, sim, zeros, ones = self.net_model.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                tops = [
                    "data", "label_muti", "label", 
                    "label_", "sim", "zeros", "ones"]
                )

        onelen = 10
        stocklen = 800
        weeklen = 7
        alllen = onelen + weeklen + onelen * stocklen
        points = [alllen, alllen * 2]
        slices = self.net_model.sliceLayer(data, points)
        self.net_model.silenceLayer(label_)
        fc1 = self.atomLayers(slices[0], label, label_muti, onelen, 1.0, is_test, ones, zeros)
        fc2 = self.atomLayers(slices[1], label, label_muti, onelen, 1.0, is_test, ones, zeros)
        fc3 = self.atomLayers(slices[2], label, label_muti, onelen, 1.0, is_test, ones, zeros)
        concat = self.net_model.concatLayer(*[fc1, fc2, fc3])
        concat = self.net_model.groupDropLayer(concat, 3, 2, False)

        fc = self.net_model.fcLayer(concat, 1024, t="xavier", 
                replace='relu')
        fc_drop1 = self.net_model.dropLayer(fc, self.dropout, False)
        """
        fc_drop2 = self.net_model.dropLayer(fc, self.dropout, False)
        fc_drop3 = self.net_model.dropLayer(fc, self.dropout, False)
        """

        out = self.net_model.fcLayer(fc_drop1, 20, t="xavier", isout=True)
        out_seg = self.net_model.sigmoidLayer(out)
        self.net_model.silenceLayer(out_seg)
        self.net_model.silenceLayer(label_muti)

        #loss = self.net_model.lossLayer(out, label_muti, "sigcross", 1)
        loss = self.net_model.lossLayer(out, label, "softmax", 1, param={"ignore_label": 0})
        acc = self.net_model.accLayer(out, label)

        """
        s1 = self.net_model.fcLayer(fc_drop1, 8, t="xavier", replace='sigmoid', wname=["s_w", "s_b"])
        s2 = self.net_model.fcLayer(fc_drop2, 8, t="xavier", replace='sigmoid', wname=["s_w", "s_b"])
        s3 = self.net_model.fcLayer(fc_drop3, 8, t="xavier", replace='sigmoid', wname=["s_w", "s_b"])
        loss_s1 = self.net_model.lossLayer(s1, s2, third_bottom=sim, t="contrastive", param={"margin": 2, "margin_sim": 2.5}, weight=0)
        loss_s2 = self.net_model.lossLayer(s1, s3, third_bottom=sim, t="contrastive", param={"margin": 2, "margin_sim": 2.5}, weight=0)
        loss_s3 = self.net_model.lossLayer(s2, s3, third_bottom=sim, t="contrastive", param={"margin": 2, "margin_sim": 2.5}, weight=0)
        """

        """
        out2 = self.net_model.fcLayer(fc, 20, t="xavier", isout=True)
        loss = self.net_model.lossLayer(out, label, "softmax", 1)

        out2_seg = self.net_model.softmaxLayer(out2)
        out_final = self.net_model.eltwiseLayer([out_seg, out2_seg], 0)

        acc = self.net_model.accLayer(out_final, label)
        """

        self.net_model.silenceLayer(sim)
        self.net_model.netStr(netname)
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

