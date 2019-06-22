#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-8-19 13:52:50$
# Note: This source file is NOT a freeware
# Version: config.py 0.1 jingyi Exp $
db = "/tran_tmp/caffedb"
config = {
    "debug": False,
    "data": {
        "data_src": "/datas/pkgs/openbrold/data/LFW/img",
        "train_dir_pre": "./caffedb",
        "train_dir": db,
        "db_size": 4000,
        "width": 227,
        "height": 227,
    },
    "runner": {
        "db": db,
        "iter_step": 1000,
        "dir": "run",
        "solver": "run/solver.prototxt",
        "model": "run/train.prototxt",
        "snapshot": "run/train_snapshot.solverstate.last",
        "brain": "run/brain.caffemodel.last",
    },
    "max_iter": 100000,
}