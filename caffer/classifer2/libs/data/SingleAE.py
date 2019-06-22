#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: SingleAE.py
# Date: 2016 Sun 13 Nov 2016 08:43:58 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import h5py
sys.path.append("/datas/lib/py")
from PyData import PyData

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class SingleAE(PyData):
    ############# rewrite start #############
    def loadX(self):
        pass

    def loadStats(self):
        pass

    ############# rewrite end #############
    def _commonConfigure(self):
        self.feat_len = 1024
        self.is_double_out = False
        self.zeros_len = 2048
        pass

    def _configure(self):
        self.batch_size = 128
        self.tops = [
                (self.batch_size, feat_len), 
                ]
        self.is_train = True
        pass

    def loadData(self):
        self.X = self.loadX()
        self.mean, self.std, self.max_n, self.min_n = self.loadStats()
        self.X_len = len(self.X)
        if not self.is_train:
            print self.X_len
        self.ignore_set = set()
        self.batch_count = 0
        self.zeros = np.zeros((self.batch_size, self.zeros_len)) + 0.0001
        self.ones = np.ones((self.batch_size, self.zeros_len))
        pass

    def batch(self):
        data = False
        target = False
        n = self.feat_len * 10
        if self.batch_count > 0 and self.batch_count % 10000 == 0:
            print "---------", len(self.ignore_set), self.batch_count
        for i in range(self.batch_size):
            self.batch_count += 1
            while True:
                rand_index = np.random.randint(0, 
                        self.X_len - self.feat_len - 1)
                if rand_index in self.ignore_set:
                    continue
                else:
                    break
            crr_info = self.X[rand_index: rand_index+self.feat_len].copy() / 10000.

            min_n = crr_info.min()
            max_n = crr_info.max()
            mean = crr_info.mean()
            std = crr_info.std()
            """
            if (max_n - min_n) >= 5 * std:
                self.ignore_set.add(rand_index)
                if len(self.ignore_set) % 10000 == 0:
                    print len(self.ignore_set)

            prev_info = self.X[rand_index-n: rand_index] / 10000.
            feat_choose = prev_info
            std = feat_choose.std()
            choose_index = self.feat_len/16 * 12
            margin = 50000
            margin = 7000
            margin_all = crr_info.max() - crr_info.min()
            #print margin, margin_all
            """

            tmp_pri = crr_info.copy()
            tmp = tmp_pri.copy()

            tmp = (tmp - min_n) * 1. / (max_n - min_n)
            tmp.shape = (1, self.feat_len)
            if not self.is_train:
                tmp_pri[-1] = tmp_pri[-2]
                tmp_pri = (tmp_pri - tmp_pri.mean()) * 1. * 255.
            else:
                tmp_pri = (tmp_pri - mean) * 1. * 255.
            tmp_pri.shape = (1, self.feat_len)
            if i == 0:
                data = tmp_pri
                target = tmp
            else:
                data = np.concatenate((
                    data, 
                    tmp_pri
                    ))
                target = np.concatenate((
                    target, 
                    tmp
                    ))
        """
        if not self.is_train:
            print [data, target]
            exit()
        assert target.max() <= 1 and target.min() >= 0, \
                "%f, %f" % (target.max(), target.min())
        print target.max(), target.min()
        print [data, target]
        exit()
        """
        return [data, target, self.zeros, self.ones]

    def loadStockH5(self, fname):
        with h5py.File(fname, 'r') as f:
            X = f['data'][:]
            y = f['label'][:] 
            y_ = f['realy'][:]

        return np.array(X).astype(np.float32), \
                np.array(y).astype(np.float32).reshape(len(y), 1, 1, 1), \
                np.array(y_).astype(np.float32)

    def testPrint(self):
        print "Hello World!"

def main():
    conf = {
            "debug": True,
            }
    t = SingleAE(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

