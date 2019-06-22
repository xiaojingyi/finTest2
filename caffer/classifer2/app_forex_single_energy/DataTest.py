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
sys.path.append("/datas/lib/py")
from DataTrain import DataTrain

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataTest(DataTrain):
    def _configure(self):
        self.is_train = True
        self.batch_size = 256
        self.tops = [
                (self.batch_size, self.feat_len), 
                (self.batch_size, self.feat_len), 
                ]
        self.is_train = True

    def loadX(self):
        X, y, y_ = self.loadStockH5("./caffe/data_1.h5")
        return X

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

