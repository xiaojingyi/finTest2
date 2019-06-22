#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataTrain.py
# Date: 2016 Mon 01 Aug 2016 10:44:41 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../libs")
from data.ForexSingle import ForexSingle

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataTrain(ForexSingle):
    def _commonConfigure(self):
        self.y_threshold = 1.0
        self.atom_num = 60
        self.onelen = self.atom_num * 4
        #self.label_padding = [0, 0, 5]
        self.y_log_scale = 10000
        self.idx_skip = self.atom_num * 5
        self.use_seperate_idx = 0.7

    def _configure(self):
        self.is_train = True
        self.batch_size = 256
        self.tops = [
                (self.batch_size, 1, 1, self.onelen), 
                (self.batch_size, 20),
                (self.batch_size, 1),
                (self.batch_size, 1),
                (self.batch_size, 1), # sim test, 0
                ]

    def batchTransform(self, res):
        res.append(np.random.randint(2, size=(self.batch_size, 1)))
        #res.append(np.ones((self.batch_size, 1)))
        return res

    def loadStockDatas(self):
        return self.loadStockH5("./caffe/data_0.h5")

    def loadMeanStd(self):
        f = "./caffe/mean.npy"
        mean = np.load(f)
        f = "./caffe/std.npy"
        std = np.load(f)

        return mean, std

def main():
    conf = {
            "debug": True,
            }
    t = DataTrain(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

