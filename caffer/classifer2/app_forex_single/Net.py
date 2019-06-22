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

    def net(self, netname, is_test):
        data, label_muti, label, label_, sim = self.net_model.pyDataLayer(
                "DataTrain" if not is_test else "DataTest", 
                "DataTrain" if not is_test else "DataTest", 
                0 if not is_test else 1,
                tops = ["data", "label_muti", "label", "label_", "sim"]
                )

        onelen = 4
        self.net_model.silenceLayer(label_)
        self.net_model.silenceLayer(label_muti)
        drop = self.net_model.groupDropLayer(data, 60, 50, False)
        conv1 = self.net_model.convLayer(drop, [onelen, 1], [onelen, 1], 64)
        layer = self.net_model.normLayer(conv1)
        conv2 = self.net_model.convLayer(layer, [3, 1], [1, 1], 
                64)
        conv3 = self.net_model.convLayer(conv2, [3, 1], [2, 1], 
                64)
        """
        """

        fc_1 = self.net_model.fcLayer(conv3, 512, t="xavier", 
                replace='relu')
        ex_drop = 0.5
        fc_drop1 = self.net_model.dropLayer(fc_1, ex_drop, False)
        fc_drop2 = self.net_model.dropLayer(fc_1, ex_drop, False)
        fc_drop3 = self.net_model.dropLayer(fc_1, ex_drop, False)

        s1 = self.net_model.fcLayer(fc_drop1, 512, t="xavier", replace='sigmoid', wname=["s_w", "s_b"])
        s2 = self.net_model.fcLayer(fc_drop2, 512, t="xavier", replace='sigmoid', wname=["s_w", "s_b"])
        s3 = self.net_model.fcLayer(fc_drop3, 512, t="xavier", replace='sigmoid', wname=["s_w", "s_b"])
        margin_min = 50
        margin_max = 100
        loss_s1 = self.net_model.lossLayer(s1, s2, third_bottom=sim, t="contrastive", param={"margin": margin_min, "margin_sim": margin_max}, weight=1)
        loss_s2 = self.net_model.lossLayer(s1, s3, third_bottom=sim, t="contrastive", param={"margin": margin_min, "margin_sim": margin_max}, weight=1)
        loss_s3 = self.net_model.lossLayer(s2, s3, third_bottom=sim, t="contrastive", param={"margin": margin_min, "margin_sim": margin_max}, weight=1)

        out = self.net_model.fcLayer(s1, 20, t="xavier", isout=True)
        out_seg = self.net_model.sigmoidLayer(out)
        self.net_model.silenceLayer(out_seg)

        loss = self.net_model.lossLayer(out, label_muti, "sigcross", 1)
        #loss = self.net_model.lossLayer(out, label, "softmax", 1)
        acc = self.net_model.accLayer(out, label)

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

