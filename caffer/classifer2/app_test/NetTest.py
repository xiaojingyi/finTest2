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
        self.net = MyCaffeNet({"debug": True})
        self.fcNet(fname, False)

    def testNet(self, fname):
        self.net = MyCaffeNet({"debug": True})
        self.fcNet(fname, True)

    def kakalaBlock(self, bottom, per=0.5):
        fc = self.net.fcLayer(bottom, 512, replace="relu")
        fc = self.net.fcLayer(fc, 512, replace="relu")
        out = self.net.binGateLayer(fc, bottom, per)
        return out

    def treeBlock(self, bottom, num=64):
        fc = self.net.fcLayer(bottom, num, replace="relu", dropout=0.5)
        out = self.net.fcLayer(fc, 20, isout=True)
        return fc, out

    def resNetMy(self, netname, is_test):
        data, label = self.net.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                )
        outs = []
        n_num = 128
        data = self.net.dropLayer(data, 0.5, in_place=False)
        fc = self.net.fcLayer(data, n_num, replace="relu", dropout=0.5)
        for i in range(10):
            fc, out = self.treeBlock(fc, n_num)
            outs.append(out)
        out = self.net.eltwiseLayer(outs, 1) # sum
        loss = self.net.lossLayer(out, label, "softmax", 1)
        acc = self.net.accLayer(out, label)

        self.net.netStr(netname)
        return

    def fcNet(self, netname, is_test):
        data, label, zeros, ones, ones_ = self.net.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                tops = ["data", "label", "zeros", "ones", "ones_"]
                )

        n_num = 128

        fc = self.net.fcLayer(data, n_num, replace="relu")
        fc_drop = fc

        # drop begin
        #fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        #fc_drop = self.net.sameDropLayer(ones, fc, False)
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 0])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 0])
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        # drop end

        fc_final = self.net.sameDropLayer(fc, fc_drop, False)
        fc_final = self.net.batchNormLayer(fc_final, gs=is_test, in_place=False)
        fc_final = self.net.scaleLayer(fc_final, in_place=False)
        fc = fc_final

        """
        fc_drop = self.net.sameDropLayer(ones_, fc_drop, False)
        fc_drop = self.net.eltwiseLayer([fc_drop, ones], opt=1)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        fc_drop = self.net.fcLayer(
                fc_drop, n_num, replace="sigmoid",
                decay=[1, 2])
        #fc_drop = self.net.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)
        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.net.sameDropLayer(fc, fc_drop, False)
        fc = self.net.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        """
        """
        fc = self.net.fcLayer(fc, n_num, replace="relu")

        loss = self.net.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        out_for_loss = self.net.sameDropLayer(fc_drop, fc, False)

        # reach 56.3%
        fc_drop = self.net.fcLayer(fc, n_num/2, replace="relu")
        fc_drop = self.net.fcLayer(fc_drop, n_num, replace="relu")
        loss = self.net.lossLayer(fc_drop, fc, "eloss", 0.1)
        # reach 56.3%

        fc_drop = self.net.fcLayer(fc, n_num, replace="sigmoid")
        fc = self.net.eltwiseLayer([fc, fc_drop], opt=0)
        loss = self.net.lossLayer(fc, zeros, "eloss", 0.1)
        fc = self.net.batchNormLayer(fc, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc = self.net.dropLayer(fc, 0.5)


        fc = fc1
        fc_drop = self.net.fcLayer(fc, n_num/2, replace="relu")
        fc_drop = self.net.dropLayer(fc_drop, 0.5)
        fc_drop = self.net.fcLayer(fc_drop, n_num, replace="relu")
        loss = self.net.lossLayer(fc_drop, fc, "eloss", 0.1)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(fc, n_num/2, replace="relu")
        fc_drop = self.net.dropLayer(fc_drop, 0.5)
        fc_drop = self.net.fcLayer(fc_drop, n_num, replace="relu")
        loss = self.net.lossLayer(fc_drop, fc, "eloss", 0.1)

        fc_plus = self.net.eltwiseLayer([fc, fc_drop], opt=1)
        loss = self.net.lossLayer(fc_plus, zeros, "eloss", 0.1)

        fc = self.net.dropLayer(fc, 0.5, in_place=False)
        fc = fc2

        fc = self.net.fcLayer(fc1_d, n_num, replace="relu")
        fc = self.net.dropLayer(fc, 0.5, in_place=False)

        fc = self.net.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.net.fcLayer(fc, n_num, replace="sigmoid")
        fc = self.net.eltwiseLayer([fc, fc_drop], opt=0)
        loss = self.net.lossLayer(fc, zeros, "eloss", 0.1)
        fc = self.net.batchNormLayer(fc, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        fc_in = self.net.fcLayer(fc, n_num, replace="sigmoid")
        fc_drop = self.net.fcLayer(fc_in, n_num, replace="sigmoid")
        fc = self.net.eltwiseLayer([fc_in, fc_drop], opt=0)
        loss = self.net.lossLayer(fc, zeros, "eloss", 0.3)
        fc = self.net.batchNormLayer(fc, in_place=False)
        fc = self.net.scaleLayer(fc, in_place=False)

        """

        self.net.silenceLayer(zeros)
        self.net.silenceLayer(ones)
        self.net.silenceLayer(ones_)
        fc = self.net.fcLayer(fc, n_num, replace="sigmoid")
        out = self.net.fcLayer(fc, 2, isout=True)
        loss = self.net.lossLayer(out, label, "softmax", 1)
        acc = self.net.accLayer(out, label)

        self.net.netStr(netname)
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

