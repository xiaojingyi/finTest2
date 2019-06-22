#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-9-3 15:03:43$
# Note: This source file is NOT a freeware
# Version: to_gray.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-9-3 15:03:43$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
from lib.Util import *
import numpy as np
import skimage
import skimage.io
import skimage.color

def main():
    dirname = sys.argv[1]
    ls = walkDir(dirname)
    for i in range(len(ls)):
        img_fname = ls[i]
        suffix = img_fname.split(".")[-1].lower()
        if suffix not in ['jpg', 'png', 'pgm']:
            continue
        img = skimage.img_as_float(skimage.io.imread(img_fname)).astype(numpy.float32)
        img = skimage.color.rgb2gray(img)
        skimage.io.imsave(img_fname, img)
        if i % 1000 == 0:
            print i, "processed"
    print "done"
    return

if __name__ == "__main__":
    main()
