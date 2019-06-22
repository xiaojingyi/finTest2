#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Predictor.py
# Date: 2016 Sat 16 Jul 2016 09:52:01 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../libs")
from PredictorBase import PredictorBase
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Predictor(PredictorBase):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Predictor init")
        self.config = config
        self.debug = config["debug"]
        super(Predictor, self).__init__(config)
        self.f_iter = 0
    
    def mkData(self):
        X, y, y_ = self.loadH5("./caffe/data_0.h5")
        Xt, yt, yt_ = self.loadH5("./caffe/data_1.h5")
        return X, y, Xt, yt, yt_

    def resTest(self, ls):
        fname="predictor_test_%d.csv" % self.f_iter
        self.f_iter += 1
        ls = sorted(ls, key=lambda x: x[0], reverse=False)
        csv_data = []
        err = {
                "good":0, 
                "total":0, 
                "profit":0,
                "potential":0,
                }

        for one in ls:
            a, s, l, yi_ = one
            if l == 1:
                err['profit'] += yi_
                if yi_ > 0:
                    err["good"] += 1
                err['total'] += 1
                err['potential'] += abs(yi_)
            elif l == 2:
                err['profit'] -= yi_
                if yi_ < 0:
                    err["good"] += 1
                err['total'] += 1
                err['potential'] += abs(yi_)
            
            if err['total'] > 0:
                a = err['good'] * 1.0 / err['total']
                b = err['profit'] / err['potential']
                csv_data.append([
                    a, b,
                    err['total'],
                    err['profit'],
                    err['potential'],
                    ])
        mkCsvFileSimple(fname, csv_data)
        print "created: ", fname
        return

    def test(self):
        self.setTData()
        batch_test = self.data_handler.loadTests()
        Xt = batch_test[0].astype(np.float32)
        yt = batch_test[1].astype(np.float32)
        yt_ = batch_test[2].astype(np.float32)
        print Xt.shape
        print yt.shape
        print yt_.shape
        self.initNet(sys.argv[1], sys.argv[2])
        res = []
        max_std = 0
        for i in range(len(Xt)):
            Xi = Xt[i]
            yi_ = yt_[i]
            res_p = self.predict(Xi)
            l = res_p[0]
            std = res_p[1]
            score = res_p[2]

            """
            a = std.mean()
            a = std[l] * (1 - score[l])
            """
            a = 1 - score[l]
            s = std[l]
            res.append([a, s, l, yi_])
            if s > max_std:
                max_std = s
            if i % 1000 == 0:
                print i

        self.resTest(res)
        ls = map(lambda x: [x[1], x[0], x[2], x[3]], res)
        self.resTest(ls)
        ls = map(lambda x: [
            x[0] + 0.1 * (x[1] / max_std), 
            x[1], x[2], x[3]], res)
        self.resTest(ls)
        ls = map(lambda x: [
            x[0] + 0.2 * (x[1] / max_std), 
            x[1], x[2], x[3]], res)
        self.resTest(ls)
        ls = map(lambda x: [
            x[0] + 0.4 * (x[1] / max_std), 
            x[1], x[2], x[3]], res)
        self.resTest(ls)

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
            "data_init_len": 500,
            "cluster_number": 3,
            "data_growth": 0.2,
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

