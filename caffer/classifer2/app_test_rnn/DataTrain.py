#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataTrain.py
# Date: 2016 Mon 01 Aug 2016 06:31:45 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../libs")
from data.PyData import PyData

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataTrain(PyData):
    def _commonConfigure(self):
        self.tlen = 20

    def _configure(self):
        self.is_train = True
        self.feat_len = 200
        self.batch_size = 128
        self.tops = [
                (self.tlen, self.batch_size, 1, 1), 
                (self.tlen, self.batch_size),
                ]

    def loadData(self):
        self.X = np.load("train_X.npy").astype(np.float32)
        self.y = np.load("train_y.npy").astype(np.float32)
        self.dlen = len(self.y)
        self.idx = np.arange(self.dlen)
        self.idx_i = 0

    def batch(self):
        if self.idx_i + self.batch_size >= self.dlen:
            self.idx_i = 0
            self.idx = np.random.permutation(self.idx)
        idx = self.idx[self.idx_i: self.idx_i+self.batch_size]

        X = self.X[idx]
        y = self.y[idx]

        self.idx_i += self.batch_size
        return [X, y]

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

