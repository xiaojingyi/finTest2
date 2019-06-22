#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: RecNet.py
# Date: 2016 2016年02月26日 星期五 20时30分31秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.MyCaffe import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class RecNet(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
    
    def splitDatas(self, bottom, dim_n, step):
        num = dim_n / step
        if dim_n % step > 0:
            num += 1
        slices = sliceLayer(bottom, dim_n, num)
        fcs = []
        for one in slices:
            tmp, top = fcLayer(one, 1, "relu")
            fcs.append(top)
        concat = concatLayer(*tuple(fcs))
        return concat, num

    def joinData(self, bottom, dim_last, dim_n):
        new_step = int(dim_n / dim_last)
        tmp = dim_n % dim_last
        tops = []
        sum_dims = 0
        for i in range(dim_last):
            if tmp > 0:
                step = new_step + 1
                tmp -= 1
            else:
                step = new_step
            fc, top = fcLayer(bottom, step, "relu")
            tops.append(top)
            sum_dims += step
        concat = concatLayer(*tuple(tops))
        return concat, sum_dims

    def net(self, dim_n):
        step = self.conf['step_num']
        data, label = dataLayer("./abs.h5", "lmdb")
        split_1, dim_num = self.splitDatas(data, dim_n, step)
        split_tmp = split_1
        while(dim_num > 4096):
            split_tmp, dim_num = self.splitDatas(split_tmp, dim_num, step)
        fc1, top = fcLayer(split_tmp, 2048, "relu", 0.5)
        fc2, top = fcLayer(top, 512, "relu", 0.3)
        fc3, top = fcLayer(top, 256, "relu", 0.3)
        fc4, top = fcLayer(top, 128, "relu")
        fc5, top = fcLayer(top, 1024, "relu")
        data2, dim_num = self.joinData(top, 1024, dim_n)
        print dim_n, dim_num
        loss = lossLayer(data2, data, "eloss", 0.4)
        saveNet("test.prototxt", loss)
        return

    def testPrint(self):
        print "Hello World!"

def main():
    conf = {"step_num": 50}
    t = RecNet(conf)
    t.net(8999000)
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

