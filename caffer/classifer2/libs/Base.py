#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Base.py
# Date: 2016 Sat 16 Jul 2016 09:20:05 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time, shutil
import json, random
import numpy as np
import math
import h5py
sys.path.append("/datas/lib/py")
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Base(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Base init")
        self.config = config
        self.debug = config["debug"]
        #super(Base, self).__init__(config)
    
    def loadOne(self, fname):
        return jsonFileLoad(fname)

    def saveOne(self, fname, data):
        return jsonFileSave(fname, data)

    def indexH5(self, fname, index):
        with h5py.File(fname, 'r') as f:
            res = f[index][:]

        return res

    def loadH5(self, fname):
        with h5py.File(fname, 'r') as f:
            X = f['data'][:]
            y = f['label'][:]
            y_ = f['realy'][:]

        return np.array(X).astype(np.float32), \
                np.array(y).astype(np.float32).reshape(len(y), 1, 1, 1), \
                np.array(y_).astype(np.float32)

    def scale(self, X, scale=255):
        return X * scale

    def transform(self, X, mean, std, scale=255):
        return self.scale((X - mean) / std, scale)

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = Base(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

