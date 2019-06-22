#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: PredictTest.py
# Date: 2016 Tue 16 Aug 2016 03:54:58 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import copy
import numpy as np
import random
sys.path.append("/datas/lib/py")
sys.path.append("/.jingyi/codes/caffer/classifer2/libs")
sys.path.append("/.jingyi/codes/caffer/dmaker")
from PredictorBase import PredictorBase
from TestCounter import TestCounter
from OneDayMaker import OneDayMaker
from OneDayMakerRT import OneDayMakerRT
from DayMaker import DayMaker
from lib.Util import *
from lib.MyMath import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class PredictTest(PredictorBase):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: PredictTest init")
        self.config = config
        self.debug = config["debug"]
        super(PredictTest, self).__init__(config)
        self.dmaker = OneDayMaker(self.config)
        self.dmaker_rt = OneDayMakerRT(self.config)
        self.rgmaker = DayMaker(self.config)
        self.placeholder = self.rgmaker.placeholder
        self.onelen = 4012
    
    def run(self):
        self.doPredict(
                sys.argv[1],
                sys.argv[2],
                self.config['dates'][0],
                self.config['dates'][1],
                "./caffe",
                )
        return

    def dataAppend(self, data, index_sh, rg, rg_wkday):
        return data, index_sh, rg, rg_wkday

    def doPredict(self, net, model, dt_from, dt_to, caffe_dir):
        data, index_sh, rg, rg_wkday = self.dataPrepare(dt_from, dt_to)
        data, index_sh, rg, rg_wkday = self.dataAppend(data, index_sh, rg, rg_wkday)

        self.getMeanStd(caffe_dir)
        self.initNet(net, model)
        self.counter = TestCounter({
            "stop_lose": 100000,
            "debug": self.debug,
            })
        res = self.predictLoop(data, index_sh, rg, rg_wkday)
        self.counter.saveRes()
        return res

    def getMeanStd(self, caffe_dir):
        f = caffe_dir + "/mean.npy"
        self.mean = np.load(f)[0: self.onelen]
        f = caffe_dir + "/std.npy"
        self.std = np.load(f)[0: self.onelen]

    def dataPrepare(self, dt_from, dt_to):
        data, index_sh = self.rgmaker.rangeDaysAll(
                dt_from,
                dt_to,
                )
        rg = self.rgmaker.time_range
        time_range_weekday = dict(
                map(lambda x: (x, weekday(x)), self.rgmaker.time_range)
                )
        return data, index_sh, rg, time_range_weekday

    def predictLoop(self, data, index_sh, rg, rg_wkday):
        rglen = len(rg)
        y_sh = []
        timelen = 2
        for i in range(rglen-timelen):
            if i == 0:
                continue
            d = rg[i]
            week_ft = [self.placeholder] * 7 
            week_ft[rg_wkday[d]] = 1 
            trend = week_ft[:]
            ys = []
            prices = []
            for j in range(self.stocklen): 
                prev_data = data[j][i-1][0:4]
                one_data = data[j][i][0:4]
                next_data = data[j][i+1][0:4]
                future_data = data[j][i+timelen][0:4]

                Xi = [prev_data[3]]
                Xi.extend(one_data[0:4])
                yi = 0
                if one_data[3] > 0 and next_data[0] > 0:
                    close = (next_data[3]-one_data[3]) / one_data[3]
                    yi = close
                    if abs(yi) > 0.13:
                        yi = 0
                if sum(one_data[0:4]) < 0.1:
                    Xi = [self.placeholder] * 5

                ys.append(yi)
                prices.extend(Xi)
                Xi = l2Norm(Xi, self.placeholder)
                trend.extend(Xi)

            #test_data = self.dmaker.caffeX(trend, prices, False)
            test_data = self.dmaker.caffeX(trend, prices, True)
            test_X = self.transform(test_data, self.mean, self.std, 255)
            y = self.predict(test_X, index="eltwise_1")
            self.processY(y, ys, self.counter)
            print "iter:", i, d
            y_sh.append(index_sh[i])
        self.counter.saveRes()
        mkCsvFileSimple("sh.csv", y_sh)
        return

    def processY(self, y, ys, counter, num=20, is_index=False):
        alpha = 0.8
        y_predict = map(lambda x: [
            x[np.argmax(x)],
            np.argmax(x),
            x[1],
            x[2],
            ], y)
        label_300 = self.labelInfo(y_predict, 300, 300, 500)
        label_800 = self.labelInfo(y_predict, int(800 * alpha), 800)
        label_all = self.labelInfo(y_predict, int(2862 * alpha))
        l = np.argmax(label_800)
        l_a = np.argmax(label_all)
        l_300 = np.argmax(label_300)
        #l_300 = 1 if label_300[1] > label_300[2] else 2
        #l_300 = 0 if label_300[1] == label_300[2] else l_300
        if is_index:
            """
            label = l
            if l != l_a:
                if l * l_a != 0:
                    label = 0
            """
            label = l_a
            print label,
            #counter.addRes(label, ys)
        else:
            res = []
            for ii in range(800):
                #if y_predict[ii][1] != 0 and ys[ii] != 0:
                if True:
                #if label_800[1] > label_800[2] \
                #        or label_all[1] > label_all[2]:
                    if y_predict[ii][1] == 1 and ys[ii] != 0:
                        res.append([
                            y_predict[ii][0],
                            y_predict[ii][1],
                            ys[ii],
                            y_predict[ii][2],
                            ])
            res = sorted(res, key=lambda x: x[0], reverse=True)
            res = map(lambda x: [x[1], x[2]], res)
            if len(res) >= 0:
                counter.addResArr(res[0:num], True)
            else:
                counter.addResArr([], True)

    def runbak(self):
        # make range data
        data, index_sh = self.rgmaker.rangeDaysAll(
                self.config['dates'][0],
                self.config['dates'][1],
                )
        rg = self.rgmaker.time_range
        time_range_weekday = dict(
                map(lambda x: (x, weekday(x)), self.rgmaker.time_range)
                )
        rglen = len(rg)
        y_sh = []
        stocklen  = 2862 # magic number from the start
        self.initNet(sys.argv[1], sys.argv[2])
        counter = TestCounter({
            "stop_lose": 100000,
            "debug": self.debug,
            })
        timelen = 2
        alpha = 0.8
        for i in range(rglen-timelen):
            if i == 0:
                continue
            d = rg[i]
            """
            trend1, _ = self.dmaker.makeData(d)
            print np.array(trend1)
            print d
            """
            week_ft = [self.rgmaker.placeholder] * 7 
            week_ft[time_range_weekday[d]] = 1 
            trend = week_ft[:]
            ys = []
            prices = []
            for j in range(stocklen): 
                prev_data = data[j][i-1][0:4]
                one_data = data[j][i][0:4]
                next_data = data[j][i+1][0:4]
                future_data = data[j][i+timelen][0:4]

                Xi = [prev_data[3]]
                Xi.extend(one_data[0:4])
                yi = 0
                if one_data[3] > 0 and next_data[0] > 0:
                    high = (next_data[1] - one_data[3]) / one_data[3]
                    low = (next_data[2] - one_data[3]) / one_data[3]
                    close = (next_data[3] - one_data[3]) / one_data[3]
                    open_sell = (next_data[0] - one_data[3]) / one_data[3]

                    """

                    open_y = (next_data[0] - one_data[0]) / one_data[3]

                    yi = (future_data[3] - one_data[3]) / one_data[3]

                    theta = 0.0
                    yi = high / 2.0 if high > theta else low
                    yi = open_sell

                    yi = (future_data[3]-next_data[0]) / next_data[0]
                    if open_y > 0.03 or open_y < -0.06:
                        yi = 0

                    up_t = 0.025
                    sell_t = -0.06
                    yi = low if low <= sell_t else \
                            up_t if high >= up_t else close
                    """
                    yi = close
                    if abs(yi) > 0.13*timelen:
                        yi = 0
                if sum(one_data[0:4]) < 0.1:
                    Xi = [self.rgmaker.placeholder] * 5

                ys.append(yi)
                """
                hd = Xi[0:1]
                bd = Xi[1:]
                bd.reverse()
                Xi = hd + bd
                """
                prices.extend(Xi)
                Xi = l2Norm(Xi, self.rgmaker.placeholder)
                trend.extend(Xi)

            #test_data = self.dmaker.caffeX(trend, prices, False)
            test_data = self.dmaker.caffeX(trend, prices, True)
            test_X = self.transform(test_data, self.mean, self.std, 255)
            models = walkDir("models")
            y = self.predict(test_X, index="eltwise_1")
            #y = self.predictMutiModel(test_X, models, index="sigmoid_1")
            y_predict = map(lambda x: [
                #random.random(),
                x[np.argmax(x)],
                #random.randint(0,2),
                np.argmax(x),

                x[1],
                x[2],
                ], y)

            label_800 = self.labelInfo(y_predict, int(800 * alpha), 800)
            label_all = self.labelInfo(y_predict, int(2862 * alpha))

            res = []
            print "iter:", i, d
            for ii in range(800):
                #if np.argmax(label_all) == 1:
                #if label_all[1] > label_all[2] \
                #        or label_800[1] > label_800[2]:
                if True:
                #    if ys[ii] != 0:
                    if y_predict[ii][1] == 1 and ys[ii] != 0:
                        res.append([
                            y_predict[ii][0],
                            y_predict[ii][1], 
                            ys[ii],
                            y_predict[ii][2], 
                            ])
            res = sorted(res, key=lambda x: x[0], reverse=True)
            res = map(lambda x: [x[1], x[2]], res)
            if len(res) >= 0:
                counter.addResArr(res[0:100], True)
            else:
                counter.addResArr([], True)
            y_sh.append(index_sh[i])
            print index_sh[i]
            """
            a = raw_input()
            """
        counter.saveRes()
        mkCsvFileSimple("sh.csv", y_sh)

    def labelInfo(self, y_predict, count, limit=2862, start=0):
        label_all = [0,0,0]
        label_stock_all = copy.deepcopy(y_predict[start:limit+start])
        label_stock_all = sorted(label_stock_all, key=lambda x: x[0], reverse=True)
        for one in label_stock_all[0:count]:
            label_all[one[1]] += 1
        return label_all

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "gpu": 0,
            "dates": ['2016-05-28', '2016-08-05'],
            #"dates": ['2016-07-12', '2016-08-03'],
            #"dates": ['2016-07-12', '2016-08-08'],
            #"dates": ['2012-01-01', '2013-01-01'],
            #"dates": ['2011-01-01', '2012-01-01'],
            #"dates": ['2010-01-01', '2011-01-01'],
            #"dates": ['2009-01-01', '2010-01-01'],
            "thread_num": 29
            }
    t = PredictTest(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

