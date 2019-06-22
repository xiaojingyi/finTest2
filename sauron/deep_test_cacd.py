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
    data_dic = {}
    person_id = 0
    ids = {}
    for i in range(len_dir):
        one = dir_ls[i]
        print one
        file_info = one.split(".")
        if file_info[-1] != "jpg":
            continue

        one_arr = one.split("_");
        one_path = img_path + '/' + one
        age = one_arr[0]
        name = "_".join(one_arr[1:-1])
        num = one_arr[1]

        from_path = one_path
        if data_dic.has_key(name):
            data_dic[name] += 1
        else:
            data_dic[name] = 1
            ids[name] = person_id
            person_id += 1

        if data_dic[name] < 3:
            to_path = "cacd_val/"+from_path
            cpWithPath(from_path, to_path)
            Xt.append(from_path)
            yt.append(ids[name])
        else:
            to_path = "cacd_train/"+from_path
            cpWithPath(from_path, to_path)
            X.append(from_path)
            y.append(ids[name])

    train_len = len(X)
    val_len = len(Xt)
    f = open("cacd_train.txt", "w")
    ft = open("cacd_val.txt", "w")
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

