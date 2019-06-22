#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Net.py
# Date: 2016 2016年04月22日 星期五 17时16分01秒
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
        #super(Net, self).__init__(config)

    def nn_sauron(self):
        net = MyCaffeNet({"debug": True})
        data, label = net.dataLayer(
                "", "mem", 10,
                tops=["data", "label"],
                memdim=[1,1,6001]
                )
        slices = net.sliceLayer(data, [3000, 6000])
        data1 = slices[0]
        data2 = slices[1]
        sim = slices[2]

        drop = 0.7
        lr = [1, 2]
        ip1 = net.fcLayer(data1, 2048, replace="relu", \
                dropout=drop, wname=["w1", "b1"], lr=lr)
        ip2 = net.fcLayer(ip1, 2048, replace="relu", \
                dropout=drop, wname=["w2", "b2"], lr=lr)
        ip3 = net.fcLayer(ip2, 128, replace="sigmoid", \
                dropout=0, wname=["w3", "b3"], lr=lr)
        decoder = net.fcLayer(ip3, 128, replace="sigmoid", wname=["dw", "db"], lr=lr)

        ip1_ = net.fcLayer(data2, 2048, replace="relu", \
                dropout=drop, wname=["w1", "b1"], lr=lr)
        ip2_ = net.fcLayer(ip1_, 2048, replace="relu", \
                dropout=drop, wname=["w2", "b2"], lr=lr)
        ip3_ = net.fcLayer(ip2_, 128, replace="sigmoid", \
                dropout=0, wname=["w3", "b3"], lr=lr)
        decoder_ = net.fcLayer(ip3_, 128, replace="sigmoid", wname=["dw", "db"], lr=lr)

        top = net.fcLayer(ip3, 10, isout=True)

        loss = net.lossLayer(top, label, "softmax", 0.1)
        loss_sim = net.lossLayer(decoder, decoder_, "contrastive", 1, third_bottom=sim)

        acc = net.accLayer(top, label)
        net.netStr("nn_sauron.prototxt")
        return

    def run(self):
        self.nn_sauron()
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
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

