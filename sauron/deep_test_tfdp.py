#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: deep_test.py
# Date: 2015 2015年06月30日 星期二 21时29分59秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.Util import *
import random

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

def main():
    X = []
    y = []
    Xt = []
    yt = []
    #img_path = os.path.abspath(sys.argv[1])
    img_path = sys.argv[1]
    dir_ls = os.listdir(img_path)
    len_dir = len(dir_ls)
    for i in range(len_dir):
        one = img_path + '/' + dir_ls[i]
        #print one
        ls = walkDir(one)
        len_one = len(ls)

        random.shuffle(ls)
        for j in range(len_one):
            from_path = ls[j]
            tmp = from_path.split(".")
            if tmp[-1] != "jpg":
                continue

            if j < int(len_one * 0.2):
                to_path = "tfdp_val/"+from_path.replace(" ", "_")
                cpWithPath(from_path, to_path)
                Xt.append(from_path)
                yt.append(i)
            else:
                to_path = "tfdp_train/"+from_path.replace(" ", "_")
                cpWithPath(from_path, to_path)
                X.append(from_path)
                y.append(i)
        print len_one, len(X), len(Xt)

    train_len = len(X)
    val_len = len(Xt)
    f = open("tfdp_train.txt", "w")
    ft = open("tfdp_val.txt", "w")
    for i in range(train_len):
        f.write("%s %d\n" % (X[i], y[i]))

    for i in range(val_len):
        ft.write("%s %d\n" % (Xt[i], yt[i]))

    f.close()
    ft.close()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

