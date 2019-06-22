#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: CaffeFeature.py
# Date: 2016 2016年03月02日 星期三 19时45分34秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
import numpy as np
from lib.Util import *
import caffe

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class CaffeFeature(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: CaffeFeature init")
        self.config = config
        self.debug = config["debug"]
        #super(CaffeFeature, self).__init__(config)

        self.deploy_prototxt = self.config['deploy_prototxt']
        self.model = self.config['model']
        self.gpu = self.config['gpu']
        if self.gpu:
            caffe.set_device(0)
            caffe.set_mode_gpu()

        self.net = caffe.Net(self.deploy_prototxt, self.model, caffe.TEST)
    
    def levelData(self, level):
        return self.net.blobs[level].data[0]

    def forward(self, data, level, dim):
        print data
        self.net.blobs['data'].data[...] = data.reshape(1, 1, 1, dim)
        print np.sum(self.net.blobs['data'].data[0] == data)
        out = self.net.forward()
        return self.levelData(level)

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = CaffeFeature(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

