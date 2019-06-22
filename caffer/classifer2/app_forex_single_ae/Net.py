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
        self.nt = MyCaffeNet({"debug": True})
        #self.convNet(fname, False)
        self.net(fname, False)
    
    def testNet(self, fname):
        self.nt = MyCaffeNet({"debug": True})
        #self.convNet(fname, True)
        self.net(fname, True)

    def convNet(self, netname, is_test):
        data, target = self.nt.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                tops = ["data", "target"]
                )
        drop, pos_mask, neg_mask = self.nt.drop2Layer(data, 0.05)
        self.nt.silenceLayer(pos_mask)

        net_dim = 1024
        drop = self.nt.reshapeLayer(drop, [0, 1, 1, net_dim])

        target = self.nt.sameDropLayer(target, neg_mask, False)
        target = self.nt.reshapeLayer(target, [0, 1, 1, net_dim])

        conv1 = self.nt.convLayer(drop,
                ksize_wh = [129, 1], 
                stride_wh = [1, 1], 
                nout = 32, 
                pad_wh = [64, 0],
                t="xavier", std=0.1, replace="")
        conv1 = self.nt.batchNormLayer(conv1, gs=is_test)
        conv1 = self.nt.scaleLayer(conv1)
        conv1 = self.nt.reluLayer(conv1)

        conv2 = self.nt.convLayer(conv1,
                ksize_wh = [129, 1], 
                stride_wh = [1, 1], 
                nout = 32, 
                pad_wh = [64, 0],
                t="xavier", std=0.1, replace="")
        conv2 = self.nt.batchNormLayer(conv2, gs=is_test)
        conv2 = self.nt.scaleLayer(conv2)
        conv2 = self.nt.reluLayer(conv2)

        conv3 = self.nt.convLayer(conv2,
                ksize_wh = [129, 1], 
                stride_wh = [1, 1], 
                nout = 32, 
                pad_wh = [64, 0],
                t="xavier", std=0.1, replace="")
        conv3 = self.nt.batchNormLayer(conv3, gs=is_test)
        conv3 = self.nt.scaleLayer(conv3)
        conv3 = self.nt.reluLayer(conv3)
        
        conv4 = self.nt.convLayer(conv3,
                ksize_wh = [129, 1], 
                stride_wh = [1, 1], 
                nout = 32, 
                pad_wh = [64, 0],
                t="xavier", std=0.1, replace="")
        conv4 = self.nt.batchNormLayer(conv4, gs=is_test)
        conv4 = self.nt.scaleLayer(conv4)
        conv4 = self.nt.reluLayer(conv4)

        conv5 = self.nt.convLayer(conv4,
                ksize_wh = [129, 1], 
                stride_wh = [1, 1], 
                nout = 32, 
                pad_wh = [64, 0],
                t="xavier", std=0.1, replace="")
        conv5 = self.nt.batchNormLayer(conv5, gs=is_test)
        conv5 = self.nt.scaleLayer(conv5)
        conv5 = self.nt.reluLayer(conv5)

        out = self.nt.deConvLayer(conv5,
                ksize_wh = [17, 1], 
                stride_wh = [1, 1], 
                pad_wh = [8, 0],
                nout = 1, 
                )

        loss = self.nt.lossLayer(
                out, target, "sigcross", 1, 
                third_bottom=neg_mask)

        sig = self.nt.sigmoidLayer(out)
        sim_target = self.nt.sameDropLayer(sig, neg_mask)
        loss = self.nt.lossLayer(sim_target, target, "eloss", 0)

        self.nt.netStr(netname)
        """
        deconv = self.nt.sigmoidLayer(deconv)

        conv = self.nt.batchNormLayer(conv, gs=is_test)
        conv = self.nt.scaleLayer(conv)
        conv = self.nt.reluLayer(conv)
        deconv = self.nt.batchNormLayer(deconv, gs=is_test)
        deconv = self.nt.scaleLayer(deconv)
        """
        return

    def net(self, netname, is_test):
        data, target, zeros, ones = self.nt.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                tops = ["data", "target", "zeros", "ones"]
                )
        drop, pos_mask, neg_mask = self.nt.drop2Layer(data, 0.3)
        target = self.nt.sameDropLayer(target, neg_mask, False)
        self.nt.silenceLayer(pos_mask)

        n_num = 2048
        fc = self.nt.fcLayer(drop, n_num, t="xavier", 
                replace='relu')

        fc = self.nt.fcLayer(fc, n_num, t="xavier", 
                replace='relu')

        fc_drop = self.nt.sameDropLayer(ones, fc, False)
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.nt.sameDropLayer(fc_drop, fc, False)
        loss = self.nt.lossLayer(out_for_loss, zeros, "eloss", 0.1)

        fc = self.nt.sameDropLayer(fc, fc_drop, False)
        fc = self.nt.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.nt.scaleLayer(fc, in_place=False)

        fc = self.nt.fcLayer(fc, n_num, t="xavier", 
                replace='relu')

        fc_drop = self.nt.sameDropLayer(ones, fc, False)
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.nt.sameDropLayer(fc_drop, fc, False)
        loss = self.nt.lossLayer(out_for_loss, zeros, "eloss", 0.1)

        fc = self.nt.sameDropLayer(fc, fc_drop, False)
        fc = self.nt.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.nt.scaleLayer(fc, in_place=False)

        fc = self.nt.fcLayer(fc, n_num, t="xavier", 
                replace='relu')

        fc_drop = self.nt.sameDropLayer(ones, fc, False)
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.nt.sameDropLayer(fc_drop, fc, False)
        loss = self.nt.lossLayer(out_for_loss, zeros, "eloss", 0.1)

        fc = self.nt.sameDropLayer(fc, fc_drop, False)
        fc = self.nt.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.nt.scaleLayer(fc, in_place=False)

        fc = self.nt.fcLayer(fc, n_num, t="xavier", 
                replace='relu')

        fc_drop = self.nt.sameDropLayer(ones, fc, False)
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.nt.sameDropLayer(fc_drop, fc, False)
        loss = self.nt.lossLayer(out_for_loss, zeros, "eloss", 0.1)

        fc = self.nt.sameDropLayer(fc, fc_drop, False)
        fc = self.nt.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.nt.scaleLayer(fc, in_place=False)

        fc = self.nt.fcLayer(fc, n_num, t="xavier", 
                replace='relu')

        fc_drop = self.nt.sameDropLayer(ones, fc, False)
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.eltwiseLayer([fc_drop, zeros], opt=1)
        out_for_loss = self.nt.sameDropLayer(fc_drop, fc, False)
        loss = self.nt.lossLayer(out_for_loss, zeros, "eloss", 0.1)

        fc = self.nt.sameDropLayer(fc, fc_drop, False)
        fc = self.nt.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.nt.scaleLayer(fc, in_place=False)

        out = self.nt.fcLayer(fc, 1024, t="xavier", 
                isout=True)

        self.nt.silenceLayer(zeros)
        self.nt.silenceLayer(ones)
        """
        """
        loss = self.nt.lossLayer(
                out, target, "sigcross", 10, 
                third_bottom=neg_mask)
        sig = self.nt.sigmoidLayer(out)

        sim_target = self.nt.sameDropLayer(sig, neg_mask, False)
        loss = self.nt.lossLayer(sim_target, target, "eloss", 0)
        """
        fc_drop = self.nt.sameDropLayer(ones, fc, False)
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        out_for_loss = self.nt.sameDropLayer(fc_drop, fc, False)
        loss = self.nt.lossLayer(out_for_loss, zeros, "eloss", 0.1)
        fc = self.nt.sameDropLayer(fc, fc_drop, False)
        fc = self.nt.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.nt.scaleLayer(fc, in_place=False)

        fc = self.nt.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.nt.scaleLayer(fc, in_place=False)

        target = self.nt.sameDropLayer(target, fc_drop, False)

        fc_drop = self.nt.fcLayer(fc, n_num, replace="sigmoid")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="sigmoid")
        fc = self.nt.eltwiseLayer([fc, fc_drop], opt=0)
        loss = self.nt.lossLayer(fc, zeros, "eloss", 0.1)
        fc = self.nt.batchNormLayer(fc, gs=is_test, in_place=False)
        fc = self.nt.scaleLayer(fc, in_place=False)

        fc = self.nt.fcLayer(fc, n_num, t="xavier", 
                replace='relu')

        fc_drop = self.nt.fcLayer(fc, n_num, replace="relu")
        fc_drop = self.nt.dropLayer(fc_drop, 0.3)
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="relu")
        loss = self.nt.lossLayer(fc_drop, fc, "eloss", 0.1)
        fc_drop = self.nt.fcLayer(fc, n_num/2, replace="relu")
        fc_drop = self.nt.fcLayer(fc_drop, n_num, replace="relu")
        loss = self.nt.lossLayer(fc_drop, fc, "eloss", 0.1)

        fc = self.nt.dropLayer(fc, 0.1)
        fc = self.nt.normLayer(fc)

        """

        self.nt.netStr(netname)
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

