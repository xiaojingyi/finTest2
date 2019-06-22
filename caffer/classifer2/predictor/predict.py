#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: predict.py
# Date: 2016 Tue 09 Aug 2016 01:03:27 AM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import copy
sys.path.append("/datas/lib/py")
sys.path.append("/.jingyi/codes/caffer/classifer2/libs")
sys.path.append("/.jingyi/codes/caffer/dmaker")
from PredictorBase import PredictorBase
from OneDayMaker import OneDayMaker
from OneDayMakerRT import OneDayMakerRT
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class predict(PredictorBase):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: predict init")
        self.config = config
        self.debug = config["debug"]
        super(predict, self).__init__(config)
        self.dmaker = OneDayMaker(self.config)
    
    def labelInfo(self, y, count, limit=2862):
        y_predict = map(lambda x: [
            x[np.argmax(x)],
            np.argmax(x)
            ], y)
        label_all = [0,0,0]
        label_stock_all = copy.deepcopy(y_predict[0:limit])
        label_stock_all = sorted(label_stock_all, key=lambda x: x[0], reverse=True)
        for one in label_stock_all[0:count]:
            label_all[one[1]] += 1
        return label_all

    def getRes(self):
        net = sys.argv[1]
        model = sys.argv[2]
        if len(sys.argv) >= 4:
            dt = sys.argv[3]
        else:
            dt = mkDate()
            self.dmaker = OneDayMakerRT(self.config)

        onelen = 4012
        f = "./caffe/mean.npy"
        mean = np.load(f)[0: onelen]
        f = "./caffe/std.npy"
        std = np.load(f)[0: onelen]

        self.initNet(net, model)

        self.dmaker.checkDate(dt)
        #X = self.dmaker.getDay(dt, skip_stops=False)
        X = self.dmaker.getDay(dt, skip_stops=True)
        #print X.sum()
        #exit()

        X = self.transform(X, mean, std, 255)
        models = walkDir("models")
        #y = self.predictMutiModel(X, models, index="sigmoid_1")
        y = self.predict(X, index="eltwise_1")
        print y.shape
        return y

    def run(self):
        y = self.getRes()
        alpha = 0.8
        up_limit = 20
        label_800 = self.labelInfo(y, int(800 * alpha), 800)
        label_all = self.labelInfo(y, int(2862 * alpha))
        print "800 label:", label_800
        print "all label:", label_all

        choose_len = 800
        res = map(lambda x: (
            x[1][np.argmax(x[1])], 
            x[0], 
            np.argmax(x[1])
            ), enumerate(y))
        res = sorted(res[0:choose_len], key=lambda x: x[0], reverse=True)
        print res[0:5]
        up_count = sell_count = 0
        csv_data = []
        for i in range(choose_len):
            one = res[i]
            if one[2] == 1:
                sid = res[i][1]
                code = self.dmaker.stockCode(sid)
                """
                if code == '600588':
                    print i, code
                """
                if code[0:3] != '300':
                    if up_count < up_limit:
                        data_one = [i, code]
                        print sid, data_one
                        csv_data.append(data_one)
                    up_count += 1
            elif one[2] == 2:
                sell_count += 1

        print "up count(%d), sell count(%d)" % (up_count, sell_count)
        mkCsvFileSimple("today.csv", csv_data)
        print "finished: ", len(res)
        return

def main():
    conf = {
            "debug": True,
            "thread_num": 29,
            }
    t = predict(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

