#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: PredictTestMerge.py
# Date: 2016 Mon 05 Sep 2016 12:55:02 AM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from PredictTest import PredictTest
from lib.MyMath import *
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class PredictTestMerge(PredictTest):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: PredictTestMerge init")
        self.config = config
        self.debug = config["debug"]
        super(PredictTestMerge, self).__init__(config)
        self.onelen = 8017
    
    def run(self):
        self.doPredict(
                sys.argv[1],
                sys.argv[2],
                self.config['dates'][0],
                self.config['dates'][1],
                "./caffe_merge",
                )
        return

    def getMeanStd(self, caffe_dir):
        f = caffe_dir + "/mean.npy"
        mean = np.load(f)
        f = caffe_dir + "/std.npy"
        std = np.load(f)
        self.mean = mean[:, 0:self.onelen]
        self.std = std[:, 0:self.onelen]
        print mean.shape, std.shape

    def predictLoop(self, data, index_sh, rg, rg_wkday):
        rglen = len(rg)
        y_sh = []
        timelen = 2
        prevs = np.zeros((self.stocklen, 3, 5)) 
        prev_y = np.zeros(self.stocklen)

        for i in range(rglen-timelen):
            if i <= 50:
                continue
            d = rg[i]
            week_ft = [self.placeholder] * 7 
            week_ft[rg_wkday[d]] = 1 
            trend_price = week_ft[:]
            trend_rsi = week_ft[:]
            trend_ave = week_ft[:]
            ys = [] # the real y
            prices = []
            for j in range(self.stocklen): 
                prev_data = data[j][i-1][0:4]
                one_data = data[j][i][0:4]
                next_data = data[j][i+1][0:4]
                future_data = data[j][i+timelen][0:4]

                res_0 = self.rgmaker.periodInfo(data[j][i:i+1])
                res_1 = self.rgmaker.periodInfo(data[j][i-5+1:i+1])
                res_2 = self.rgmaker.periodInfo(data[j][i-10+1:i+1])
                res_3 = self.rgmaker.periodInfo(data[j][i-20+1:i+1])
                res_4 = self.rgmaker.periodInfo(data[j][i-40+1:i+1])
                rsi = [ 
                        res_0['rsi'],
                        res_1['rsi'],
                        res_2['rsi'],
                        res_3['rsi'],
                        res_4['rsi'],
                        ]   
                ma = [ 
                        res_0['ave_close'],
                        res_1['ave_close'],
                        res_2['ave_close'],
                        res_3['ave_close'],
                        res_4['ave_close'],
                        ]  
                price = [prev_data[3]]
                price.extend(one_data[0:4])
                prices.extend(price)
                yi = 0
                if one_data[3] > 0 and next_data[0] > 0:
                    close = (next_data[3]-one_data[3]) / one_data[3]
                    yi = close
                    """
                    lose = (next_data[2]-one_data[3]) / one_data[3]
                    high = (next_data[1]-one_data[3]) / one_data[3]
                    stop_lose = 0.03
                    if close < -stop_lose:
                        yi = -stop_lose
                    elif close < high - stop_lose:
                        yi = high - stop_lose 
                    else:
                        yi = close
                    if abs(yi) > 0.13:
                        yi = 0
                    """
                Xi = [self.placeholder] * 5
                Xi = l2Norm(Xi, self.placeholder)
                if yi == 0:
                    # dx
                    trend_price.extend(Xi)
                    trend_ave.extend(Xi)
                    trend_rsi.extend(Xi)
                    # x
                    trend_price.extend(Xi)
                    trend_ave.extend(Xi)
                    trend_rsi.extend(Xi)
                else:
                    if prev_y[j] == 0:
                        trend_price.extend(Xi)
                        trend_ave.extend(Xi)
                        trend_rsi.extend(Xi)
                    else:
                        prev = prevs[j]
                        dprice = np.array(price) - np.array(prev[0])
                        drsi = np.array(rsi) - np.array(prev[1])
                        dma = np.array(ma) - np.array(prev[2])
                        dprice = l2Norm(dprice, self.placeholder)
                        drsi = l2Norm(drsi, self.placeholder)
                        dma = l2Norm(dma, self.placeholder)
                        trend_price.extend(dprice)
                        trend_rsi.extend(drsi)
                        trend_ave.extend(dma)
                    prevs[j] = np.array([
                        price, rsi, ma
                        ])
                    price = l2Norm(price, self.placeholder)
                    rsi = l2Norm(rsi, self.placeholder)
                    ma = l2Norm(ma, self.placeholder)
                    trend_price.extend(price)
                    trend_rsi.extend(rsi)
                    trend_ave.extend(ma)
                prev_y[j] = yi
                # now skip the today up than 9.8%
                if prev_data[3] <= self.placeholder:
                    yi = 0
                elif prev_data[3] > self.placeholder \
                        and yi != 0 \
                        and abs(one_data[3] - prev_data[3]) / prev_data[3] > 0.098:
                            yi = 0

                ys.append(yi)
            sh_y_i = index_sh[i+1][3] - index_sh[i][3]

            price_data = self.toCaffeX(trend_price, 10, prices)
            price_data = self.transform(
                    price_data, self.mean[0], self.std[0], 255)
            rsi_data = self.toCaffeX(trend_rsi, 10, prices)
            rsi_data = self.transform(
                    rsi_data, self.mean[1], self.std[1], 255)
            ave_data = self.toCaffeX(trend_ave, 10, prices)
            ave_data = self.transform(
                    ave_data, self.mean[2], self.std[2], 255)
            test_X = np.concatenate(
                    (price_data, rsi_data, ave_data), 
                    axis=3)
            y = self.predict(test_X, index="sigmoid_4")
            self.processY(y, ys, self.counter, 5)
            #self.counter.stop_lose = 30
            #self.processY(y, sh_y_i, self.counter, is_index=True)
            print ", iter:", i, d
            y_sh.append(index_sh[i])
        self.counter.saveRes()
        mkCsvFileSimple("sh.csv", y_sh)
        return


def main():
    conf = {
            "debug": True,
            "gpu": 0,
            #"dates": ['2016-04-01', '2016-10-01'],
            #"dates": ['2016-04-01', '2016-09-14'],
            #"dates": ['2016-05-28', '2016-08-05'],
            #"dates": ['2016-07-12', '2016-08-03'],
            #"dates": ['2016-07-12', '2016-08-08'],
            "dates": ['2009-01-01', '2013-01-01'],
            #"dates": ['2012-01-01', '2013-01-01'],
            #"dates": ['2011-01-01', '2012-01-01'],
            #"dates": ['2010-01-01', '2011-01-01'],
            #"dates": ['2009-01-01', '2010-01-01'],
            "thread_num": 29
            }
    t = PredictTestMerge(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

