#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: TrainerCmdTest.py
# Date: 2016 2016年02月29日 星期一 11时27分40秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import random
sys.path.append("/datas/lib/py")
sys.path.append("../lib")
from CaffeTrainerCmd import CaffeTrainerCmd
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class TrainerCmdTest(CaffeTrainerCmd):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: TrainerCmdTest init")
        self.config = config
        self.debug = config["debug"]
        super(TrainerCmdTest, self).__init__(config)
        self.loadMatrix()
        return
    
    def loadMatrix(self):
        datas = jsonFileLoad("user_matrix.json")
        self.users = []
        for k in datas.keys():
            self.users.append(datas[k])
        return

    def dataSrc(self, i, bsize):
        print bsize
        X = np.zeros((bsize, 900000))
        u = self.users[i * bsize: (i+1) * bsize]
        u_len = len(u)
        for j in range(bsize):
            if j < u_len:
                for one in u[j]:
                    X[j][one] = 1
        y = np.zeros((bsize, 1))
        return X, y

    def dataSrcTest(self):
        return self.dataSrc(0, 10)

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": False,
            "test_iter": 10,
            "cache_dir": "train_test_cache",
            "model_prototxt": "train_val.prototxt",
            "solver_prototxt_template": "solver.template",
            "pre_model": "",
            "batch_size": 1500,
            "train_batch_size": 100,
            "data_gen_dir": "/home/caffe/",
            "data_use_dir": "/tmp/caffe_train/",
            "h5_file": "trainer_cmd",
            "max_iter": 10000,
            "data_len": 299377,
            "review_time": 2,
            }
    t = TrainerCmdTest(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

