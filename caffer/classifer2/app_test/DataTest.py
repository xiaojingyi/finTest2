#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataTest.py
# Date: 2016 Mon 01 Aug 2016 06:58:26 PM CST
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
        pass

    def loadData(self):
        self.X = np.load("train_Xt.npy").astype(np.float32)
        self.y = np.load("train_yt.npy").astype(np.float32)
        self.dlen = len(self.y)
        self.idx = np.arange(self.dlen)
        self.idx_i = 0

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

