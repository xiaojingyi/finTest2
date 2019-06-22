#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: ModelCutter.py
# Date: 2016 Fri 12 Aug 2016 11:45:24 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import caffe
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class ModelCutter(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: ModelCutter init")
        self.config = config
        self.debug = config["debug"]
        #super(ModelCutter, self).__init__(config)
    
    def run(self):
        proto = sys.argv[1]
        model = sys.argv[2]
        self.net = caffe.Net(proto, str(model), caffe.TRAIN)
        self.net.save("m_0_iter_6629000.caffemodel")
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
    t = ModelCutter(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

