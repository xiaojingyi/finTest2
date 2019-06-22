#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DayMakerMerge.py
# Date: 2016 Mon 22 Aug 2016 05:52:56 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import h5py
sys.path.append("/datas/lib/py")
from classes.DMaker import DMaker
from lib.Util import *
from lib.MyMath import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DayMakerMerge(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DayMakerMerge init")
        self.config = config
        self.debug = config["debug"]
        super(DayMakerMerge, self).__init__(config)
    
    def run(self):
        data, index_sh = self.rangeDaysAll(
                self.config['dates'][0],
                self.config['dates'][1],
                )
        rg = self.time_range
        time_range_weekday = dict(
                map(lambda x: (x, weekday(x)), self.time_range)
                )
        rglen = len(rg)
        X = []
        Xt = []
        y = []
        yt = []
        y_sh = []
        print "stock len:", self.stocklen
        onelen = 5
        prevs = np.zeros((self.stocklen, 3, 5))
        prev_y = np.zeros(self.stocklen)
        for i in range(rglen-2):
            if i < 50:
                continue
            d = rg[i]
            if d > self.config["trdates"][1] \
                    and d > self.config["tdates"][1]:
                        break
            print "making", d
            week_ft = [self.placeholder] * 7 
            week_ft[time_range_weekday[d]] = 1 
            trend_price = week_ft[:]
            trend_ave = week_ft[:]
            trend_rsi = week_ft[:]
            ys = []
            yst = []
            for j in range(self.stocklen): 
                prev_data = data[j][i-1][0:4]
                amount = data[j][i-1][5]

                one_data = data[j][i][0:4]

                next_data = data[j][i+1][0:4]
                next2_data = data[j][i+2][0:4]

                # period info
                price = [prev_data[3]]
                price.extend(one_data[0:4])
                res_0 = self.periodInfo(data[j][i:i+1])
                res_1 = self.periodInfo(data[j][i-5+1:i+1])
                res_2 = self.periodInfo(data[j][i-10+1:i+1])
                res_3 = self.periodInfo(data[j][i-20+1:i+1])
                res_4 = self.periodInfo(data[j][i-40+1:i+1])
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
                #Xi.extend(one_data[0:4])
                #print Xi
                yi = yit = 0
                y_thres = 0.13
                if next_data[0] > 0 and one_data[3] > 0:
                    #yi = (next2_data[3] - next_data[0]) / next_data[0]
                    """
                    if next_data[3] - one_data[3] > 0:
                        yi = next_data[1] - one_data[3]
                    else:
                        yi = next_data[2] - one_data[3]
                    yi = yi / one_data[3]
                    """
                    yi = (next_data[3] - one_data[3])*1.0 / one_data[3]
                    if abs(yi) > y_thres:
                        yi = 0
                    """
                    open_lago = close_lago = 0
                    if yi > 0:
                        open_lago = y_thres - ((one_data[3] - next_data[2])*1.0 / one_data[3])
                        close_lago = y_thres - ((next_data[1] - next_data[3])*1.0 / one_data[3])
                    elif yi < 0:
                        open_lago = y_thres - ((next_data[1] - one_data[3])*1.0 / one_data[3])
                        close_lago = y_thres - ((next_data[3] - next_data[2])*1.0 / one_data[3])
                    yi = yi * open_lago * close_lago
                    """

                    yit = (next_data[3] - one_data[3])*1.0 / one_data[3]
                    if abs(yit) > y_thres:
                        yit = 0

                if sum(one_data[0:4]) < 0.1:
                    yi = yit = 0
                Xi = [self.placeholder] * (onelen)
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
                        trend_ave.extend(drsi)
                        trend_rsi.extend(dma)
                    prevs[j] = np.array([
                        price, rsi, ma
                        ])
                    price = l2Norm(price, self.placeholder)
                    rsi = l2Norm(rsi, self.placeholder)
                    ma = l2Norm(ma, self.placeholder)
                    trend_price.extend(price)
                    trend_ave.extend(rsi)
                    trend_rsi.extend(ma)
                prev_y[j] = yi
                ys.append(yi)
                yst.append(yit)
            
            # index sh data
            sh_y_i = index_sh[i+1][3] - index_sh[i][3]
            if d >= self.config["tdates"][0] and d <= self.config['tdates'][1]:
                Xt.append([trend_price, trend_ave, trend_rsi])
                yt.extend(yst)
                y_sh.append(sh_y_i)
            elif d >= self.config["trdates"][0] and d <= self.config['trdates'][1]:
                X.append([trend_price, trend_ave, trend_rsi])
                y.extend(ys)
        X = np.array(X).astype(np.float32)
        y_ = np.array(y).astype(np.float32)
        Xt = np.array(Xt).astype(np.float32)
        yt_ = np.array(yt).astype(np.float32)
        print "allX shape:", X.shape
        print "oneX shape:", X[:, 0].shape
        mean = np.zeros((X.shape[1], X.shape[2]+10)) 
        std = np.zeros((X.shape[1], X.shape[2]+10))
        for i in range (X.shape[1]):
            mean[i], std[i] = \
                    self.getMeans(X[:, i], Xt, 
                            onelen=onelen*2, hlen=7)
        for j in range(std.shape[0]):
            for i in range(std.shape[1]):
                one = std[j][i]
                if one == 0:
                    std[j][i] = self.placeholder
                    print 'zero:', i
        print "mean:", mean
        print "std:", std
        print X.shape, Xt.shape, mean.shape, std.shape
        y = np.array(map(lambda x: self.labelMaker(x), y))
        print self.counter(y)
        yt = np.array(map(lambda x: self.labelMaker(x), yt))
        print self.counter(yt)
        # now save
        print X.shape, Xt.shape, y.shape, yt.shape

        np.save(self.caffe_dir+"/mean.npy", mean)
        np.save(self.caffe_dir+"/std.npy", std)

        self.saveH5DB(X, y, y_, False)
        self.saveH5DB(Xt, yt, yt_, False)

        fname = self.caffe_dir+"/y_sh.h5"
        with h5py.File(fname, 'w') as f:
            f["y_sh"] = np.array(y_sh).astype(np.float32)
        return 
    
def main():
    conf = {
            "debug": True,
            "thread_num": 26,

            # current pro data
            "dates": ['2008-01-01', '2016-05-28'],
            "trdates": ['2013-01-02', '2016-05-28'],
            "tdates": ['2010-01-01', '2013-01-01'],

            # quick test: 
            #"dates": ['2012-01-01', '2013-01-01'],
            #"trdates": ['2012-03-28', '2012-06-05'],
            #"tdates": ['2012-06-07', '2012-08-02'],

            ##### test: 2008-01-01 - 2016-08-19
            #"dates": ['2008-01-01', '2016-08-19'],
            #"trdates": ['2013-01-02', '2016-08-19'],
            #"tdates": ['2010-01-01', '2013-01-01'],
            }
    t = DayMakerMerge(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

