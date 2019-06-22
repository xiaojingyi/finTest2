#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: StockSigleRnn.py
# Date: 2016 Fri 12 Aug 2016 06:05:31 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from StockSigle import StockSigle

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class StockSigleRnn(StockSigle):
    ############# rewrite start #############
    def _configure(self):
        pass

    def loadStockDatas(self):
        pass

    def loadMeanStd(self):
        pass

    ############# public start #############
    def batch(self):
        t = time.time()

        XX = np.zeros((self.tlen, self.stream_len, 1, self.onelen))
        yy = np.zeros((self.tlen, self.stream_len))
        yy_ = np.zeros((self.tlen, self.stream_len))
        yy_muti = np.zeros((self.tlen, self.stream_len, 1, 20))
        t_ = time.time()
        #print t_ - t
        t = time.time()
        for ii in range(self.stream_len):
            y_muti = self.y_zeros * 0
            stock_id = np.random.randint(0, self.stock_len)
            rows_start = np.random.randint(0, self.daylen - self.tlen)

            # make X
            Xi = self.X_base[
                    rows_start: rows_start+self.tlen,
                    :, :,
                    stock_id * 4: (stock_id+1) * 4
                    ]
            X = np.concatenate((Xi, self.trend[
                rows_start: rows_start+self.tlen
                ]), axis=3)

            # make y
            y = self.y[rows_start: rows_start+self.tlen, 0, 0, stock_id]
            y_ = self.y_[rows_start: rows_start+self.tlen, 0, 0, stock_id]
            #y_ = y_ - y_[0]

            X = self.transform(X, self.mean, self.std, 255)
            #print X.shape, y.shape, y_.shape, y_muti.shape

            # make multi-y
            for i in range(self.tlen):
                l = y[i]
                walk = y_[i]
                if l == 0:
                    y_muti[i][0][0][0] = self.label_mean
                else:
                    y_muti[i][0][0][l] = abs(walk)

            XX[:, ii] = X.reshape((self.tlen, 1, self.onelen))
            yy[:, ii] = y.reshape((y.shape[0]))
            yy_[:, ii] = y_.reshape((y.shape[0]))
            yy_muti[:, ii] = y_muti.reshape((self.tlen, 1, 20))
        blen = self.tlen * self.stream_len
        XX.shape = (blen, 1, 1, self.onelen)
        yy.shape = (blen)
        yy_.shape = (blen)
        yy_muti.shape = (blen, 1, 1, 20)
        """
        print XX
        print yy
        print yy_
        print yy_muti
        print self.clips

        t_ = time.time()
        print t_ - t
        exit()
        """
        return [XX, yy_muti, yy, yy_, self.clips]

    def afterLoadData(self):
        self.stock_len = 2862

        if self.is_train:
            self.y_ = self.normY_(self.y_)

        # make trend matrix
        self.trend = self.X_base[:, :, :, 0:800*4]
        self.y = self.y.reshape(self.X_base.shape[0], 1, 1, -1)
        self.y_ = self.y_.reshape(self.X_base.shape[0], 1, 1, -1)
        assert self.y.shape[-1] == self.stock_len
        assert self.y_.shape[-1] == self.stock_len
        
        self.daylen = self.X_base.shape[0]
        self.clips = np.ones((self.tlen, self.stream_len, 1))
        self.clips[0] *= 0
        self.clips.shape = (self.tlen * self.stream_len, 1)
        self.y_zeros = np.zeros((self.tlen, 1, 1, 20))

        #y_abs = abs(self.y_).max(0) + 0.00000001
        #self.y_ = self.y_ / y_abs
        """
        print self.y_, self.y_.max(), self.y_.min()
        exit()
        print y_abs.shape
        for one in y_abs[0][0]:
            if one == 0:
                print "ddd"

        assert_data = self.y_[100].copy()
        print self.y_
        print self.y_
        assert_data2 = self.y_[100] - self.y_[99]
        print (assert_data2 - assert_data).sum()
        exit()

        for i in range(self.daylen):
            if i < 1:
                continue
            self.y_[i] = self.y_[i-1:i+1].sum(0)
        """

def main():
    conf = {
            "debug": True,
            }
    t = StockSigleRnn(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

