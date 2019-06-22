#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataFactory.py
# Date: 2016 2016年05月07日 星期六 17时22分50秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataFactory(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DataFactory init")
        self.config = config
        self.debug = config["debug"]
        #super(DataFactory, self).__init__(config)
    
    def init(self, mname):
        lib = __import__("datas." + mname)
        #print dir(lib)
        classes = getattr(lib, mname)
        model = getattr(classes, mname)
        #print dir(model)
        obj = model(self.config)
        return obj

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = DataFactory(conf)
    handler = t.init("mem")
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

