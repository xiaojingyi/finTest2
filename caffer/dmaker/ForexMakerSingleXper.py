#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: ForexMakerSingle.py
# Date: 2016 Fri 16 Sep 2016 10:04:07 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from lib.Util import *
from lib.MyMath import *
from lib.threadpool import ThreadPool
from classes.DMaker import DMaker

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class ForexMakerSingle(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: ForexMakerSingle init")
        self.config = config
        self.debug = config["debug"]
        # we just use few methods of the parent
        super(ForexMakerSingle, self).__init__(config)

    def loadCsv(self, fname):
        lines = readFileLines(fname)
        len_lines = len(lines)
        res = np.zeros((len_lines, 4))
        for i in range(len_lines):
            one = lines[i].split(",")
            res[i][0] = float(one[2]) * 10000
            res[i][1] = float(one[3]) * 10000
            res[i][2] = float(one[4]) * 10000
            res[i][3] = float(one[5]) * 10000
        return res

    def mkTrainTest(self, data, split_t=0.2):
        feat_num = data.shape[0]

        #X_base = []
        X_base = np.zeros(feat_num)
        y = np.zeros(feat_num)
        for i in range(feat_num):
            # no y
            y[i] = 0
            if i % 1000000 == 0:
                print "iter:", i
            if i == 0:
                X_base[i] = (data[i][3] - data[i][0])
            else:
                X_base[i] = (data[i][3] - data[i-1][3])
        """
        X_base = abs(X_base)
        X_base *= 200.
        X_base = sigmoid(X_base)
            if data[i][3] > 0:
                X_base.append(data[i][3])
        """
        #X_base = np.array(X_base)
        #X_base = X_base[-200000:]
        #X_base *= 50
        print X_base
        print X_base.shape
        print X_base.max(), X_base.min()
        print X_base.std(), X_base.mean()
        split_point = int(X_base.shape[-1]*(1-split_t))

        X_base_tr = X_base[0:split_point]
        y_tr_ = y[0:split_point]
        y_tr = y_tr_

        X_base_t = X_base[split_point:]
        y_t_ = y[split_point:]
        y_t = y_t_

        # now save
        print "start saving..."
        stats = np.array([
            X_base.mean(),
            X_base.std(),
            X_base.max(),
            X_base.min(),
            ])
        np.save(self.caffe_dir+"/stats.npy", stats)

        print X_base_tr.shape, X_base_t.shape, y_tr.shape, y_t.shape
        self.saveH5DB(X_base_tr, y_tr, y_tr_, False)
        self.saveH5DB(X_base_t, y_t, y_t_, False)

    def run(self):
        # load the csv
        data = self.loadCsv("./EURUSD1.csv")
        print "data ok"
        self.mkTrainTest(data)

    def __del__(self):
        pass
    
def main():
    conf = {
            "debug": True,
            }
    t = ForexMakerSingle(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

