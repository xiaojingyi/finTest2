#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataTest.py
# Date: 2016 2016年06月04日 星期六 22时27分14秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
from Data import Data
import numpy as np
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataTest(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataTest init")
        self.config = config
        self.debug = config["debug"]
        super(DataTest, self).__init__(config)
    
    def initData(self):
        self.batch_len = self.config["batch_len"]
        self.is_sim = self.config["is_sim"]
        self.dcache_prefix = self.config["dcache_prefix"]
        self.data_growth = self.config['data_growth']
        self.dlen_use = self.config['data_init_len']
        return

    def placeAllData(self, X, y, Xt, yt, yt_=[]):
        print X.shape, Xt.shape
        self.X = X
        self.y = y
        self.Xt = Xt
        self.yt = yt
        self.yt_ = yt_
        self.dlen = len(y)
        self.dtlen = len(yt)
        return len(y), len(yt)

    def loadBaches(self):
        shape = list(self.X.shape)
        shape[-1] = 1
        shape = tuple(shape)
        X = np.concatenate((self.X, np.ones(shape)), axis=3)
        return [X, self.y]

    def loadTests(self):
        shape = list(self.Xt.shape)
        shape[-1] = 1
        shape = tuple(shape)
        X = np.concatenate((self.Xt, np.ones(shape)), axis=3)
        return [X, self.yt, self.y]

    def useDataLen(self):
        return self.dlen

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

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

