#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: SingleEnergy.py
# Date: 2016 Sun 20 Nov 2016 03:21:14 PM CST
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

class SingleEnergy(PyData):
    ############# rewrite start #############
    def loadX(self):
        pass

    ############# rewrite end #############
    def _commonConfigure(self):
        self.feat_len = 1024
        self.is_double_out = False
        pass

    def _configure(self):
        self.batch_size = 128
        self.tops = [
                (self.batch_size, feat_len), # changed data 
                (self.batch_size, feat_len), # pri data
                ]
        self.is_train = True
        pass

    def loadData(self):
        self.X = self.loadX()
        self.X_len = len(self.X)
        self.ener = np.arange(self.feat_len) * 1.0 / self.feat_len 
        pass

    def batch(self):
        data = False
        pridata = False
        for i in range(self.batch_size):
            rand_index = np.random.randint(0, 
                    self.X_len - self.feat_len - 1)
            tmp = self.X[rand_index: rand_index+self.feat_len]
            tmpas = tmp.argsort()
            final = tmp[:].copy()
            final[tmpas] = self.ener
            
            tmp.shape = (1, self.feat_len)
            final.shape = (1, self.feat_len)
            if i == 0:
                data = final
                pridata = tmp
            else:
                data = np.concatenate((data, final))
                pridata = np.concatenate((pridata, tmp))

        """
        print data
        print pridata
        exit()
        """
        return [data, pridata]

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

