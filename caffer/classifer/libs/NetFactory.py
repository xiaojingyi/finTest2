#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: NetFactory.py
# Date: 2016 2016年05月08日 星期日 19时05分04秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class NetFactory(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: NetFactory init")
        self.config = config
        self.debug = config["debug"]
        #super(NetFactory, self).__init__(config)
    
    def init(self, mname, param, fname="net.prototxt"):
        lib = __import__("nets." + mname)
        #print dir(lib)
        classes = getattr(lib, mname)
        model = getattr(classes, mname)
        #print dir(model)
        net = model(self.config)
        net.run(fname, param)
        print "created net."
        return fname

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = NetFactory(conf)
    t.testPrint()
    t.init("Net", {})
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

