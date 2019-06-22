#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-8-19 1:40:20$
# Note: This source file is NOT a freeware
# Version: Temp.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-8-19 1:40:20$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
from lib.Util import *
import numpy as np

class Temp(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Temp init")
        self.config = config
        self.debug = config["debug"]
        #super(Temp, self).__init__(config)
        
    def mkSolver(self, i, step):
        fp = open("template/solver.prototxt", "r")
        content = fp.read()
        fp.close()
        content = content.replace("{max_iter}", str((i+1) * step))
        writeToFile("run/solver.prototxt", content)
        return
    
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()
        
    def test(self):
        print "debug: ", self.debug
        
def main():
    config = {
        "debug": True,
    }
    model = Temp(config)
    model.test()
    return

if __name__ == "__main__":
    main()
