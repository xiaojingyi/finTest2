#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: CaffePreprocess.py
# Date: 2015 2015年07月18日 星期六 14时32分50秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class CaffePreprocess(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
    
    def lmdbCmd(self, preffix):
        content = "convert_imageset --resize_height=227 --resize_width=227 --shuffle ./ %s.txt %s_lmdb " % (preffix, preffix)
        writeToFile("%s_lmdb.sh" % (preffix), content)
    
    def meanCmd(self, preffix):
        content = "compute_image_mean %s_lmdb %s_mean.binaryproto" % (preffix, preffix)
        writeToFile("%s_mean.sh" % (preffix), content)
        
    def fileLs(self, preffix):
        file_ls = walkDir(preffix)
        content = ""
        for one in file_ls:
            cat = one.split(".")[-2]
            content += "%s %s\n" % (one, cat)
        writeToFile("%s.txt" % (preffix), content)
        
    def run(self, set_path):
        set_path = set_path.strip('/')
        self.fileLs(set_path)
        self.lmdbCmd(set_path)
        self.meanCmd(set_path)
        
    def testPrint(self):
        print "Hello World!"

def main():
    conf = {}
    t = CaffePreprocess(conf)
    t.run(sys.argv[1])
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

