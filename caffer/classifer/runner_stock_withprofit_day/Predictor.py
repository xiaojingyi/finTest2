#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Predictor.py
# Date: 2016 Tue 26 Jul 2016 05:07:11 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time, random
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
        self.last_label = 0
    
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
            print i
            idx = idx_all[i*bsize:i*bsize+bsize]
            y = self.data_handler.yt[idx]
            y_ = self.data_handler.yt_[idx]
            X, y = self.data_handler.loadX(
                    self.data_handler.Xt_base, 
                    idx, bsize, y, False)
            #res = self.predict(X, usesig=False)
            res = self.predict(X, usesig=True)
            #res = res[0:base_size]
            #l = self.outCalc(res, 1)
            #counter.addRes(l, self.index_sh[i], is_calc=True)

            # test only up
            data = self.upCalc(i, res, X, y_, bsize, 1000)
            counter.addResArr(data, all_in_one=False)

        counter.saveRes()

        return 

    def upCalc(self, iter_i, out, X, y_, bsize, dlen=50):
        ls = []
        for i in range(bsize):
            one = out[i]
            yi_ = y_[i]
            xi = X[i]
            l = np.argmax(one)
            # random a/b test
            #l = random.randint(0,2)
            #one[l] = random.random()
            ls.append([one[l], int(l), yi_, xi[0][0][0:5]])
        ls = sorted(ls, key=lambda x: x[0], reverse=True)

        print dlen
        res = []
        len_ct = 0
        profit_sum = 0
        counter = [0,0,0]
        for i in range(dlen):
            one = ls[i]
            counter[one[1]] += 1
            """
            if i < 300:
                continue
            if one[1] == 1:
                print one[3], [one[1], one[2]]
                profit_sum += one[2]
                res.append([one[1], one[2]])
                len_ct += 1
                if len_ct > 10:
                    break

            """
            len_ct += 1
            res.append([one[1], one[2]])
        res = map(lambda x: [x[0], x[1]], res)
        #res = map(lambda x: [x[0], x[1]/len_ct], res)
        print counter

        """
        label = np.argmax(counter)
        if label != 0 and self.last_label != 0:
            if label != self.last_label:
                label = 0
        #res = [ [ label, self.index_sh[iter_i] ] ]
        self.last_label = 0
        """
        print "sum:", profit_sum
        return res

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
            "gpu": 1,
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

