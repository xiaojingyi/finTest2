#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataTest.py
# Date: 2016 Mon 01 Aug 2016 10:52:05 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from DataTrain import DataTrain

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataTest(DataTrain):
    def _configure(self):
        self.is_train = False
        self.batch_size = 32
        self.tops = [
                (self.batch_size, 1, 1, self.onelen*3), 
                (self.batch_size, 20),
                (self.batch_size, 1),
                (self.batch_size, 1),
                (self.batch_size, 1), # the sim
                (self.batch_size, self.hidden_len),
                (self.batch_size, self.hidden_len),
                ]
        self.label_mean = 1
        self.zeros = np.zeros((self.batch_size, self.hidden_len))+0.00000001
        self.ones = np.ones((self.batch_size, self.hidden_len))

    def loadStockDatas(self):
        return self.loadStockH5("./caffe/data_1.h5")

def main():
    conf = {
            "debug": True,
            }
    t = DataTest(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

