#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-9-15 18:08:45$
# Note: This source file is NOT a freeware
# Version: db_show.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-9-15 18:08:45$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
import numpy as np
import h5py

def main():
    file=h5py.File(sys.argv[1],'r')
    x = file['data'][:]
    y = file['label'][:]
    file.close()
    print x
    print y
    return

if __name__ == "__main__":
    main()
