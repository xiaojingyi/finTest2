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
        data, pridata = self.nt.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                tops = ["data", "pridata"]
                )
        self.nt.silenceLayer(pridata)
        data_use, pos_mask, neg_mask = self.nt.drop2Layer(data, 0.1)
        target = self.nt.sameDropLayer(data, neg_mask)

        data_use = self.nt.reshapeLayer(data_use, [0, 1, 2, 1024])
        target = self.nt.reshapeLayer(target, [0, 1, 2, 1024])

        conv = self.nt.convLayer(data_use,
                ksize_wh = [32, 2], 
                stride_wh = [4, 1], 
                nout = 64, 
                pad_wh = [2, 1],
                t="xavier", std=0.1, replace="relu")
        pool = self.nt.poolLayer(conv,
                ksize_wh = [3, 3], 
                stride_wh = [1, 1], 
                t = "max", 
                )
        pool = self.nt.dropLayer(pool, 0.5)
        """
        conv = self.nt.convLayer(pool,
                ksize_wh = [5, 1], 
                stride_wh = [1, 1], 
                nout = 64, 
                pad_wh = [2, 0],
                t="xavier", replace="sigmoid")
        deconv = self.nt.deConvLayer(conv,
                ksize_wh = [5, 1], 
                stride_wh = [1, 1], 
                pad_wh = [2, 0],
                nout = 128, 
                )
        deconv = self.nt.sigmoidLayer(deconv)

        conv = self.nt.batchNormLayer(conv, gs=is_test)
        conv = self.nt.scaleLayer(conv)
        conv = self.nt.reluLayer(conv)
        deconv = self.nt.batchNormLayer(deconv, gs=is_test)
        deconv = self.nt.scaleLayer(deconv)
        """
        unpool = self.nt.unpoolLayer(pool,
                ksize_wh = [3, 3], 
                stride_wh = [1, 1], 
                nout = 64
                )
        deconv = self.nt.deConvLayer(unpool,
                ksize_wh = [32, 2], 
                stride_wh = [4, 1], 
                pad_wh = [2, 1],
                nout = 1, 
                )
        deconv = self.nt.reluLayer(deconv)
        out = self.nt.deConvLayer(deconv,
                ksize_wh = [17, 1], 
                stride_wh = [1, 1], 
                pad_wh = [8, 0],
                nout = 1, 
                )

        loss = self.nt.lossLayer(
                out, target, "sigcross", 1, 
                third_bottom=neg_masks)

        sig = self.nt.sigmoidLayer(out)
        sim_target = self.nt.sameDropLayer(sig, neg_masks)
        loss = self.nt.lossLayer(sim_target, target, "eloss", 0)

        self.nt.netStr(netname)

    def net(self, netname, is_test):
        data, pridata = self.nt.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                tops = ["data", "pridata"]
                )
        self.nt.silenceLayer(pridata)
        data_use, pos_mask, neg_mask = self.nt.drop2Layer(data, 0.1)
        self.nt.silenceLayer(pos_mask)
        target = self.nt.sameDropLayer(data, neg_mask)

        fc = self.nt.fcLayer(data_use, 2048, t="xavier", 
                replace='relu')
        fc = self.nt.fcLayer(fc, 2048, t="xavier", 
                replace='relu')
        fc = self.nt.fcLayer(fc, 2048, t="xavier", 
                replace='relu')
        out = self.nt.fcLayer(fc, 1024, t="xavier", 
                isout=True)

        loss = self.nt.lossLayer(
                out, target, "sigcross", 1, 
                third_bottom=neg_mask)
        sig = self.nt.sigmoidLayer(out)
        sim_target = self.nt.sameDropLayer(sig, neg_mask)
        loss = self.nt.lossLayer(sim_target, target, "eloss", 0)
        """
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

