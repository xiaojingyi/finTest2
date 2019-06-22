#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Data.py
# Date: 2016 2016年05月21日 星期六 15时30分39秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Data(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Data init")
        self.config = config
        self.debug = config["debug"]
        #super(Data, self).__init__(config)
        self.initData()
    
    def placeAllData(self, X, y, Xt, yt, yt_=[]):
        return

    def loadBaches(self):
        return

    def loadOrderBatchs(self, offset, length): # ignore steps
        return

    def loadTests(self):
        return

    def useDataLen(self): # data increment
        return

    # methods optional start
    def initData(self):
        return

    def addDataSet(self): # data increment
        return True

    def isMax(self): # data increment
        return True

    def stopData(self): # muti-threading
        return

    def ignore(self, i, y): # ignore steps used
        return

    # private methods start
    def simData(self, X, X_, y, y_):
        d_len = len(X)
        shape_len = len(X.shape)
        #print X.shape
        assert d_len == len(y)

        sim = np.array(map(lambda i: [int(y[i] == y_[i])], range(d_len)))
        y_new = y
        """
        y_new = np.array(map(
            lambda i: y[i] * self.config["cluster_number"] + y_[i], 
            range(d_len)
            ))
        """
        s = list(X.shape)
        s[-1] = 1
        sim.shape = tuple(s)

        X_final = np.concatenate((X, X_, sim), shape_len-1)
        return X_final, y_new

    def transform(self, X, mean, std, scale=255):
        X = X - mean
        X /= std
        X = X * scale
        return X

    def pad(self, data, t=256):
        len_data = len(data)
        n = t - len_data % t
        data = np.concatenate((data, data[0:n]))
        return data

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = Data(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

