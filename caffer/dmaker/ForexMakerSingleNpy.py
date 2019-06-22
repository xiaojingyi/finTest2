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
            res[i][0] = float(one[2])
            res[i][1] = float(one[3])
            res[i][2] = float(one[4])
            res[i][3] = float(one[5])
        return res

    def mkTrainTest(self, data, split_t=0.2):
        feat_num = data.shape[0]

        X_base = np.array(data)
        print X_base.shape
        split_point = int(feat_num*(1-split_t))

        X_train = X_base[0:split_point]
        X_test = X_base[split_point:]
        print X_train.shape
        print X_test.shape
        np.save(self.caffe_dir+"/forex-train.npy", X_train)
        np.save(self.caffe_dir+"/forex-test.npy", X_test)

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

