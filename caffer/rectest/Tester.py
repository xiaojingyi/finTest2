#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Tester.py
# Date: 2016 2016年03月02日 星期三 19时54分20秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../lib")
from CaffeFeature import CaffeFeature
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Tester(CaffeFeature):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Tester init")
        self.config = config
        self.debug = config["debug"]
        super(Tester, self).__init__(config)
        self.loadMatrix()
        self.rank = 0
        self.rank_ct = 0
        return
    
    def loadMatrix(self):
        datas = jsonFileLoad("user_matrix.json")
        self.users = []
        for k in datas.keys():
            self.users.append(datas[k])
        return
    
    def oneDataV2(self, i):
        X = np.zeros((2000))
        if i > -1:
            for one in self.users[i]:
                tmp = int(one)
                a1000 = int(tmp / 1000) 
                X[a1000] = 1
                b1000 = tmp % 1000 + 1000
                X[b1000] = 1
        return X

    def oneDataV1(self, i):
        X = np.zeros((900000))
        if i > -1:
            for one in self.users[i]:
                X[one] = 10
        return X

    def testV2(self, i):
        print "test", i
        data = self.oneDataV2(i)
        arg_mat1 = np.argwhere(data > 0.5)
        arg_mat1 = arg_mat1.reshape(len(arg_mat1))
        print arg_mat1
        data1 = self.forward(data, "Sigmoid2", 2000)
        print data1
        print len(data1)
        print "max: %f" % np.max(data1)
        print "min: %f" % np.min(data1)
        print "mean: %f" % np.mean(data1)
        args_mat = np.argsort(data1).tolist()
        args_mat.reverse()
        orders = {}
        for i in range(len(args_mat)):
            orders[args_mat[i]] = i
        for one in arg_mat1:
            print orders[one], data1[one]
            self.rank += orders[one]
            self.rank_ct += 1

    def test(self, i):
        data = self.oneDataV1(i)
        arg_mat1 = np.argwhere(data > 0.5)
        arg_mat1 = arg_mat1.reshape(len(arg_mat1))
        print arg_mat1
        data1 = self.forward(data, "Sigmoid1", 900000)
        print data1
        print "max: %f" % np.max(data1)
        print "min: %f" % np.min(data1)
        print "mean: %f" % np.mean(data1)
        args_mat = np.argsort(data1).tolist()
        args_mat.reverse()
        orders = {}
        for i in range(len(args_mat)):
            orders[args_mat[i]] = i
        for one in arg_mat1:
            print orders[one], data1[one]
            self.rank += orders[one]
            self.rank_ct += 1

    def run(self):
        for i in range(10):
            self.test(i)
        self.test(-1)
        print self.rank
        print self.rank_ct
        print self.rank / self.rank_ct
        """
        """
        return

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "deploy_prototxt": sys.argv[1],
            "model": sys.argv[2],
            "gpu": False,
            }
    t = Tester(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

