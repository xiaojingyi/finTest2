#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-9-11 23:13:18$
# Note: This source file is NOT a freeware
# Version: kill.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-9-11 23:13:18$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
import numpy as np
import random
from lib.Util import *

def main():
    ret = exeCmd("ps -ef | grep main.py")
    tmp = ret.split('\n')
    for k in tmp:
        if k.find("python")>0:
            ret = k
    arr = ret.split(" ")
#    print arr
    index = 0
    for one in arr:
        if len(one) > 0:
            index += 1
            if index == 2:
                cmd = "kill -9 %s" % one
                exeCmd(cmd)
                print cmd
                cmd = "killall caffe"
                exeCmd(cmd)
                print cmd
                cmd = "rm -rf cache/* "
                if len(sys.argv) > 1 and sys.argv[1] == 'all':
                    exeCmd(cmd)
                print cmd
                cmd = "rm /tran_tmp/caffedb*  -f"
                if len(sys.argv) > 1 and sys.argv[1] == 'all':
                    exeCmd(cmd)
                print cmd
    return

if __name__ == "__main__":
    main()
