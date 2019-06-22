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
from data.SingleEnergy import SingleEnergy

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataTrain(SingleEnergy):
    def _commonConfigure(self):
        self.feat_len = 1024
        self.is_double_out = True

    def _configure(self):
        self.is_train = True
        self.batch_size = 256
        self.tops = [
                (self.batch_size, self.feat_len), 
                (self.batch_size, self.feat_len), 
                ]
        self.is_train = True

    def loadX(self):
        X, y, y_ = self.loadStockH5("./caffe/data_0.h5")
        print X.shape
        print X
        return X

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

