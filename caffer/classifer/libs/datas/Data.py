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
import threading
import math
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Data(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Data init")
        self.config = config
        self.debug = config["debug"]
        #super(Data, self).__init__(config)
        self.placeholder = 0.00001
        self.bzeros = np.array([])
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

    def norm(self, v):
        r = np.array(v)
        d = math.sqrt(r.dot(r))
        if d == 0:
            r *= 0
        else:
            r = r * 1.0 / d
        return r

    def balancePer(self, y):
        label = {}
        for i in y:
            l = str(int(i))
            if label.has_key(l):
                label[l] += 1.0
            else:
                label[l] = 1.0
        if label['1'] > label['2']:
            return 1, label['2'] / label['1']
        else:
            return 2, label['1'] / label['2']

    def scale(self, X, scale=255):
        return X * scale

    def transform(self, X, mean, std, scale=255):
        return self.scale((X - mean) / std, scale)

    def _loadX(self, Xs, idx, blen, swl, y=[], y_=[], is_test=False):
        ii = 0
        t = time.time()
        if len(y) > 0:
            assert len(y_) > 0
            muti_label = True
            y_profit = np.zeros((blen, 1, 1, 20))
        else:
            muti_label = False
            y_profit = False
        s = swl[0] # one len
        w = swl[1] # week len
        l = swl[2] # trend len
        l_ = swl[3] # stock len

        if self.bzeros.shape[0] != blen:
            if muti_label:
                self.bzeros = np.ones((blen, 1, 1, 20+w+s*(1+l)))
            else:
                self.bzeros = np.ones((blen, 1, 1, w+s*(1+l)))
        X = self.bzeros
        
        """
        t_ = time.time()
        print t_ - t, ii
        ii += 1
        t = t_

        """

        X_index = 0
        for one_i in idx:
            # make Xi
            i = one_i / l_
            j = one_i % l_
            patten = Xs[i][0][0][w+j*s : w+j*s+s]
            week = Xs[i][0][0][0 : w]
            trend = Xs[i][0][0][w : w+s*l]
            one_X = self.transform(
                    np.concatenate( (patten, week, trend) ),
                    self.mean, self.std, 255
                    )

            # make labels
            # NOTE: must have self.label_mean!!!
            if muti_label:
                tmp = abs(y_[X_index]).reshape(1)
                if y[X_index] == 0:
                    y_profit[X_index][0][0][0] = self.label_mean
                else:
                    y_profit[X_index][0][0][int(y[X_index])] = tmp
                X[X_index][0][0] = np.concatenate(
                        (y_profit[X_index][0][0], one_X)
                        )
            else:
                X[X_index][0][0] = one_X
            X_index += 1

        X = X.astype(np.float32)
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

