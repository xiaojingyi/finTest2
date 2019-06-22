#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: create.py
# Date: 2016 Mon 01 Aug 2016 06:43:32 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import numpy as np
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

def main():
    X, y = sklearn.datasets.make_classification(
            n_samples=10000, n_features=1000, 
            n_redundant=0, n_informative=30,
            n_classes=3, n_clusters_per_class=2, 
            hypercube=False, random_state=0
            )
    X, Xt, y, yt = train_test_split(X, y, test_size=0.2)

    fname = "train_X.npy"
    np.save(fname, X)
    fname = "train_Xt.npy"
    np.save(fname, Xt)

    fname = "train_y.npy"
    np.save(fname, y)
    fname = "train_yt.npy"
    np.save(fname, yt)
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

