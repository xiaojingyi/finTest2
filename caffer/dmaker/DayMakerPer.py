#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DayMakerPer.py
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

class DayMakerPer(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DayMakerPer init")
        self.config = config
        self.debug = config["debug"]
        super(DayMakerPer, self).__init__(config)
    
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
        volume_ema5 = np.zeros(self.stocklen)
        volume_ema10 = np.zeros(self.stocklen)
        volume_ema20 = np.zeros(self.stocklen)
        close_ema5 = np.zeros(self.stocklen)
        close_ema10 = np.zeros(self.stocklen)
        close_ema20 = np.zeros(self.stocklen)
        close_ema40 = np.zeros(self.stocklen)
        for i in range(rglen-2):
            if i < 1:
                continue
            d = rg[i]
            print "making", d
            week_ft = [self.placeholder] * 7 
            week_ft[time_range_weekday[d]] = 1 
            trend = week_ft[:]
            ys = []
            yst = []
            for j in range(self.stocklen): 
                prev_data = data[j][i-1][0:4]
                amount = data[j][i-1][5]

                one_data = data[j][i][0:4]
                volume = data[j][i][4]
                close_price = one_data[3]

                next_data = data[j][i+1][0:4]
                next2_data = data[j][i+2][0:4]

                dv_5 = dv_10 = dv_20 = dc_5 \
                        = dc_10 = dc_20 = self.placeholder
                # ema
                if volume > 0:
                    volume_ema5[j], dv_5 = emaStock(volume_ema5[j], volume, 5)
                    volume_ema10[j], dv_10 = emaStock(volume_ema10[j], volume, 10)
                    volume_ema20[j], dv_20 = emaStock(volume_ema20[j], volume, 20)
                if close_price > 0:
                    close_ema5[j], dc_5 = emaStock(close_ema5[j], close_price, 3)
                    close_ema10[j], dc_10 = emaStock(close_ema10[j], close_price, 6)
                    close_ema20[j], dc_20 = emaStock(close_ema20[j], close_price, 12)
                    close_ema40[j], dc_40 = emaStock(close_ema40[j], close_price, 24)

                Xi = [
                        one_data[3], 
                        close_ema5[j],
                        close_ema10[j],
                        close_ema20[j],
                        close_ema40[j],
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
                    open_lago = close_lago = 0
                    if yi > 0:
                        open_lago = y_thres - ((one_data[3] - next_data[2])*1.0 / one_data[3])
                        close_lago = y_thres - ((next_data[1] - next_data[3])*1.0 / one_data[3])
                    elif yi < 0:
                        open_lago = y_thres - ((next_data[1] - one_data[3])*1.0 / one_data[3])
                        close_lago = y_thres - ((next_data[3] - next_data[2])*1.0 / one_data[3])
                    yi = yi * open_lago * close_lago

                    yit = (next_data[3] - one_data[3])*1.0 / one_data[3]
                    if abs(yit) > 0.13:
                        yit = 0

                #Xi = map(lambda x: x * volume, Xi)
                if sum(one_data[0:4]) < 0.1:
                    yi = yit = 0
                if yi == 0:
                    Xi = [self.placeholder] * (onelen)

                ys.append(yi)
                yst.append(yit)
                Xi = l2Norm(Xi, self.placeholder)
                trend.extend(Xi)
            
            # index sh data
            sh_y_i = index_sh[i+1][3] - index_sh[i][3]
            if d >= self.config["tdates"][0] and d <= self.config['tdates'][1]:
                Xt.append(trend)
                yt.extend(yst)
                y_sh.append(sh_y_i)
            elif d >= self.config["trdates"][0] and d <= self.config['trdates'][1]:
                X.append(trend)
                y.extend(ys)
        X = np.array(X).astype(np.float32)
        y_ = np.array(y).astype(np.float32)
        Xt = np.array(Xt).astype(np.float32)
        yt_ = np.array(yt).astype(np.float32)
        mean, std = self.getMeans(X, Xt, onelen=onelen, hlen=7)
        for i in range(len(std)):
            one = std[i]
            if one == 0:
                std[i] = self.placeholder
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

            ##### test: 2008-01-01 - 2016-08-19
            #"dates": ['2008-01-01', '2016-08-19'],
            #"trdates": ['2013-01-02', '2016-08-19'],
            #"tdates": ['2010-01-01', '2013-01-01'],
            }
    t = DayMakerPer(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

