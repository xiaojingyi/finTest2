#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: NetBuilder.py
# Date: 2016 2016年02月26日 星期五 20时30分31秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.MyCaffe import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class NetBuilder(object):
    def __init__(self, config):
        self.config = {}
        self.config = config
    
    def splitDatas(self, bottom, dim_n, step, data=False, split_dim=10):
        num = dim_n / step
        if dim_n % step > 0:
            num += 1
        slices, top_dims = sliceLayer(bottom, dim_n, num)
        print len(slices), dim_n, step
        fcs = []
        for one in slices:
            tmp, top = fcLayer(one, split_dim, "relu")
            fcs.append(top)
        concat = concatLayer(*tuple(fcs))

        if data:
            slice_data, top_dims = sliceLayer(data['data'], data['dim'], num) 
            for i in range(len(slice_data)):
                fc_tmp, top = fcLayer(fcs[i], top_dims[i])
                loss = lossLayer(top, slice_data[i], "eloss", data['loss'])
                self.losses.append(loss)

        return concat, num * split_dim

    def joinData(self, bottom, dim_last, dim_n, split_dim=10):
        assert(dim_last % split_dim == 0)

        slice_dim = dim_last / split_dim
        new_step = int(dim_n / slice_dim)
        tmp = dim_n % slice_dim
        tops = []
        sum_dims = 0
        slices, top_dims = sliceLayer(bottom, dim_last, slice_dim)
        for i in range(slice_dim):
            if tmp > 0:
                step = new_step + 1
                tmp -= 1
            else:
                step = new_step
            fc, top = fcLayer(slices[i], step, std=0.0001675)
            tops.append(top)
            sum_dims += step
        concat = concatLayer(*tuple(tops))
        return concat, sum_dims

    def net(self, dim_n, net_name="test.prototxt"):
        step = self.config['step_num']
        data, label = dataLayer("./abs.h5", "lmdb")
        split_1, dim_num = self.splitDatas(data, dim_n, step)
        split_tmp = split_1
        while(dim_num > 4096):
            split_tmp, dim_num = self.splitDatas(split_tmp, dim_num, step)

        last_dim = 4096
        fc1, top = fcLayer(split_tmp, last_dim, "relu", 0.2)
        fc1, top = fcLayer(top, last_dim, "relu", 0.2)
        fc1, top = fcLayer(top, last_dim, "relu", 0.2)
        #data3, dim_num = self.joinData(top, last_dim, dim_n)
        """
        data4, dim_num = self.joinData(top, last_dim, dim_n)
        """
        last_dim = 4096
        fc1, top = fcLayer(top, last_dim, "relu", 0.2)
        fc1, top = fcLayer(top, last_dim, "relu", 0.2)
        fc1, top = fcLayer(top, last_dim, "relu", 0.2)
        last_dim = 1000
        fc1, top = fcLayer(top, last_dim, "relu")
        data2, dim_num = self.joinData(top, last_dim, dim_n)
        print dim_n, dim_num
        sigmoid = sigmoidLayer(data2)
        losses = []
        losses.append(lossLayer(data, sigmoid, "eloss", 1))
        #losses.append(lossLayer(data, data3, "eloss", 0.3))
        #losses.append(lossLayer(data, data4, "eloss", 0.3))
        losses.append(lossLayer(label, label, "eloss", 0))
        #losses.append(lossLayer(data, data2, "sigcross", 1))
        saveNet(net_name, losses)
        return

    def testPrint(self):
        print "Hello World!"

def main():
    config = {"step_num": 50}
    t = NetBuilder(config)
    t.net(8999000)
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

