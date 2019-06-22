#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-9-13 22:01:10$
# Note: This source file is NOT a freeware
# Version: status.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-9-13 22:01:10$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
import numpy as np
import random
from lib.Util import *

def main():
    cmd = "tail -n 50 test.log"
    print exeCmd(cmd)
    cmd = "tail -n 50 train.log"
    print exeCmd(cmd)
    iter = np.load("cache/global_iter.npy")
    print "--------------iteration--------------"
    print iter
    return

if __name__ == "__main__":
    main()
