#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: npydiff.py
# Date: 2016 Tue 30 Aug 2016 02:36:44 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class npydiff(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: npydiff init")
        self.config = config
        self.debug = config["debug"]
        #super(npydiff, self).__init__(config)
    
    def testPrint(self):
        a = np.load(sys.argv[1])
        b = np.load(sys.argv[2])
        diff = a - b
        print diff.mean(), diff.sum(), diff.std()

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = npydiff(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

