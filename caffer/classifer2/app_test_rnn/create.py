#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: create.py
# Date: 2016 Fri 12 Aug 2016 12:31:04 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import h5py
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class create(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: create init")
        self.config = config
        self.debug = config["debug"]
        #super(create, self).__init__(config)
    
    def data(self, dlen, tlen, num_f, num_t, bsize):
        y = np.zeros((dlen, tlen))
        for i in range (dlen):
            rand_num = np.random.randint(num_f, num_t-tlen)
            y[i] = np.arange(rand_num, rand_num+tlen)
        X = y.copy()
        X[:, 1:] = 1
        clips = np.ones(y.shape)
        clips[:, 0] = 0
        y = y.reshape(bsize, dlen * tlen / bsize)
        X = X.reshape(bsize, dlen * tlen / bsize)
        clips = clips.reshape(bsize, dlen * tlen / bsize)
        y = y.transpose().astype(np.float32)
        X = X.transpose().astype(np.float32)
        clips = clips.transpose().astype(np.float32)
        shape = X.shape
        print y
        print X
        print clips
        print X.shape
        print y.shape
        print clips.shape
        return X.reshape((shape[0], shape[1], 1, 1)), y, clips

    def run(self):
        tlen = 32
        X, y, clips = self.data(100000, tlen, 1, 100, 200)
        fname = "train.h5"
        with h5py.File(fname, 'w') as f:
            f['data'] = X
            f['label'] = y
            f['clips'] = clips
        with open("train.txt", 'w') as f:
            f.write(fname + '\n')

        X, y, clips = self.data(1000, tlen, 0, 100, 200)
        fname = "test.h5"
        with h5py.File(fname, 'w') as f:
            f['data'] = X
            f['label'] = y
            f['clips'] = clips
        with open("test.txt", 'w') as f:
            f.write(fname + '\n')
        return

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = create(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

