#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DayMakerAERL.py
# Date: 2016 Thu 22 Dec 2016 11:23:33 PM CST
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

class DayMakerAERL(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DayMakerAERL init")
        self.config = config
        self.debug = config["debug"]
        super(DayMakerAERL, self).__init__(config)
    
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
        Xf = []
        Xt = []
        Xft = []
        y = []
        yt = []
        y_ = []
        yt_ = []
        y_sh = []
        print "stock len:", self.stocklen
        onelen = 5
        for i in range(rglen-2):
            if i < 1:
                continue
            d = rg[i]
            print "making", d
            week_ft = [self.placeholder] * 7 
            week_ft[time_range_weekday[d]] = 1 
            trend = week_ft[:]
            trend_ = week_ft[:]
            ys = []
            ys_ = []
            for j in range(self.stocklen): 
                prev_data = data[j][i-1][0:4]
                one_data = data[j][i][0:4]
                next_data = data[j][i+1][0:4]
                close_price = one_data[3]

                Xi = [prev_data[3]]
                Xfi = [one_data[3]]
                Xi.extend(one_data[0:4])
                Xfi.extend(next_data[0:4])
                #print Xi
                yi = 0
                y_thres = 0.13
                if next_data[0] > 0 and one_data[3] > 0:
                    #yi = (next_data[1] - one_data[3]
                    #        + next_data[2] - one_data[3])
                    #yi = yi * 0.5 / one_data[3]
                    yi = next_data[3] - one_data[3]
                    yi = yi * 1. / one_data[3]
                    if abs(yi) > y_thres:
                        yi = 0

                if sum(one_data[0:4]) < 0.1:
                    yi = 0
                if yi == 0:
                    Xi = [self.placeholder] * (onelen)

                yi_ = Xi[-1] - Xi[0] 
                yi_ = 0 if (yi_ == 0 or Xi[0] == 0) \
                        else (yi_ * 1. / Xi[0])
                if abs(yi_) > y_thres:
                    yi_ = 0

                ys.append(yi)
                ys_.append(yi_)
                Xi = firstNorm(Xi)
                trend.extend(Xi)
                Xfi = firstNorm(Xfi)
                trend_.extend(Xfi)
            
            # index sh data
            sh_y_i = (index_sh[i+1][3] - index_sh[i][3]) / (index_sh[i][3] + 0.000001)
            print index_sh[i], sh_y_i
            if d >= self.config["tdates"][0] and d <= self.config['tdates'][1]:
                Xt.append(trend)
                Xft.append(trend_)
                yt.append(ys)
                yt_.append(ys_)
                y_sh.append(sh_y_i)
            elif d >= self.config["trdates"][0] and d <= self.config['trdates'][1]:
                X.append(trend)
                Xf.append(trend_)
                y.append(ys)
                y_.append(ys_)
        X = np.array(X).astype(np.float32)
        Xf = np.array(Xf).astype(np.float32)
        y = np.array(y).astype(np.float32)
        y_ = np.array(y_).astype(np.float32)
        Xt = np.array(Xt).astype(np.float32)
        Xft = np.array(Xft).astype(np.float32)
        yt = np.array(yt).astype(np.float32)
        yt_ = np.array(yt_).astype(np.float32)
        mean, std = self.getMeans(X, Xt, onelen=onelen, hlen=7)
        for i in range(len(std)):
            one = std[i]
            if one == 0:
                std[i] = self.placeholder
                print 'zero:', i
        print "mean:", mean
        print "std:", std
        print X.shape, Xt.shape, mean.shape, std.shape
        # now save
        print X.shape, Xt.shape, y.shape, yt.shape

        np.save(self.caffe_dir+"/mean.npy", mean)
        np.save(self.caffe_dir+"/std.npy", std)

        # X the input with 5 price, y the next day walk, y_ now walk
        self.saveH5DB(X, Xf, y, y_, False)
        self.saveH5DB(Xt, Xft, yt, yt_, False)

        fname = self.caffe_dir+"/y_sh.h5"
        with h5py.File(fname, 'w') as f:
            f["y_sh"] = np.array(y_sh).astype(np.float32)
        return 
    
def main():
    conf = {
            "debug": True,
            "thread_num": 26,

            # current pro data
            #"dates": ['2010-01-01', '2017-07-21'],
            #"trdates": ['2010-01-02', '2016-12-27'],
            #"tdates": ['2017-01-01', '2017-07-21'],

            # old pro data
            #"dates": ['2008-01-01', '2017-05-08'],
            #"trdates": ['2013-01-02', '2017-05-08'],
            #"tdates": ['2010-01-01', '2013-01-01'],

            ##### test: 2017-01-01 - 2017-06-01
            "dates": ['2017-03-01', '2017-07-20'],
            "trdates": ['2017-03-02', '2017-05-08'],
            "tdates": ['2017-05-09', '2017-07-20'],
            }
    t = DayMakerAERL(conf)
    t.run()
    return

if __name__ == "__main__":
    main()


# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

