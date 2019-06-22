#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: ForexMakerSingle.py
# Date: 2016 Fri 16 Sep 2016 10:04:07 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from lib.Util import *
from lib.MyMath import *
from lib.threadpool import ThreadPool
from classes.DMaker import DMaker

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class ForexMakerSingle(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: ForexMakerSingle init")
        self.config = config
        self.debug = config["debug"]
        # we just use few methods of the parent
        super(ForexMakerSingle, self).__init__(config)

    def loadCsv(self, fname):
        lines = readFileLines(fname)
        len_lines = len(lines)
        res = np.zeros((len_lines, 4))
        for i in range(len_lines):
            one = lines[i].split(",")
            res[i][0] = float(one[2]) * 10000
            res[i][1] = float(one[3]) * 10000
            res[i][2] = float(one[4]) * 10000
            res[i][3] = float(one[5]) * 10000
        return res

    #######################################
    # feat_step: the skip size when making datas
    # the time range is: feat_len * feat_step
    def mkTrainTest(self, data, split_t=0.2, 
            feat_len=60, feat_step=5):
        feat_num = data.shape[0] - feat_len * feat_step - 1
        X_base = np.zeros((feat_num, data.shape[1]))
        y = np.zeros(feat_num)
        alpha = 0.95
        for i in range(feat_num):
            if i == 0:
                X_base[i] = data[i] - 10000
            else:
                X_base[i] = X_base[i-1] + data[i] - 10000
            # for y
            y_ = 0
            for j in range(feat_len):
                y_ += (data[i+j+1][3] - data[i+j][3]) * (alpha ** j) 
            """
            close = 0
            if data[i][3] > 0:
                close = (data[i+feat_len*1+1][3]-data[i][3])\
                        / data[i][3]
            y_ = close
            """
            y[i] = y_
            if i % 1000000 == 0:
                print "iter:", i
        print X_base
        print X_base.shape
        print y
        print y.max(), y.min(), y.mean(), y.std()
        split_point = int(feat_num*(1-split_t))

        X_base_tr = X_base[0:split_point]
        y_tr_ = y[0:split_point]

        X_base_t = X_base[split_point-feat_len*feat_step:]
        y_t_ = y[split_point-feat_len*feat_step:]

        y_tr = np.array(map(lambda x: self.labelMaker(x), y_tr_))
        print self.counter(y_tr)
        y_t = np.array(map(lambda x: self.labelMaker(x), y_t_))
        print self.counter(y_t)
        # now save
        print "start saving..."
        print X_base_tr.shape, X_base_t.shape, y_tr.shape, y_t.shape
        self.saveH5DB(X_base_tr, y_tr, y_tr_, False)
        self.saveH5DB(X_base_t, y_t, y_t_, False)

        print "calc std and mean..."
        mean, std = self.getMeanStd(X_base_tr, feat_len, feat_step)
        print mean.shape, std.shape
        np.save(self.caffe_dir+"/mean.npy", mean)
        np.save(self.caffe_dir+"/std.npy", std)

    def getMeanStd(self, X_base, feat_len, feat_step):
        mean = None
        std = None
        for i in range(feat_len):
            X = np.zeros((X_base.shape)) + self.placeholder
            for j in range(X_base.shape[0]):
                step = i * feat_step
                if j <= step:
                    pass
                else:
                    X[j] = (X_base[j] - X_base[j-step-1]) / (step+1)
                #X[j] = l2Norm(X[j], self.placeholder)
            print X
            mean_ = X.mean(0)
            std_ = X.std(0)
            print mean_, std_
            if i == 0:
                mean = mean_
                std = std_
            else:
                mean = np.concatenate((mean, mean_))
                std = np.concatenate((std, std_))
            print i, mean.shape, std.shape
        return mean, std

    def run(self):
        # load the csv
        data = self.loadCsv("./EURUSD1.csv")
        print "data ok"
        self.mkTrainTest(data)

    def __del__(self):
        pass
    
def main():
    conf = {
            "debug": True,
            }
    t = ForexMakerSingle(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

