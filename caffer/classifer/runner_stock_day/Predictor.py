#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Predictor.py
# Date: 2016 Tue 26 Jul 2016 05:07:11 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../libs")
from PredictorBase import PredictorBase
from TestCounter import TestCounter
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Predictor(PredictorBase):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Predictor init")
        self.config = config
        self.debug = config["debug"]
        super(Predictor, self).__init__(config)
    
    def mkData(self):
        X, y, y_ = self.loadH5("./caffe/data_0.h5")
        Xt, yt, yt_ = self.loadH5("./caffe/data_1.h5")
        self.index_sh = self.indexH5("./caffe/y_sh.h5", 'y_sh')
        return X, y, Xt, yt, yt_

    def test(self):
        self.setTData()
        idx_all = self.data_handler.data_idx_t
        dlen = len(idx_all)
        bsize = 2862
        base_size = 800
        assert dlen % bsize == 0

        print dlen, bsize
        print self.index_sh.shape
        loop_max = dlen / bsize

        self.initNet(sys.argv[1], sys.argv[2])
        counter = TestCounter({
            "debug": self.debug,
            "fname": "test_res.csv",
            "stop_lose": 30,
            "crr_len": 30,
            })
        counter.init()
        for i in range (loop_max):
            idx = idx_all[i*bsize:i*bsize+bsize]
            y = self.data_handler.yt[idx]
            X, y = self.data_handler.loadX(
                    self.data_handler.Xt_base, 
                    idx, bsize, y, False)
            res = self.predict(X, usesig=True)
            res = res[0:base_size]
            l = self.outCalc(res, 0.65)
            counter.addRes(l, self.index_sh[i])
        counter.saveRes()

        return 

    def run(self):
        self.test()

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "data_init_len": 80000,
            "cluster_number": 3,
            "data_growth": 0.8,
            "data_shape": (1, 1, 4012), #TODO
            "batch_len": 128 * 1,
            "dcache_prefix": "alex",
            "data_type": "DataStockRange",
            "is_sim": False,
            }
    t = Predictor(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

