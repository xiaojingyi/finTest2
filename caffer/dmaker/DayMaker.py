#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DayMaker.py
# Date: 2016 Sun 07 Aug 2016 07:46:51 PM CST
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

class DayMaker(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DayMaker init")
        self.config = config
        self.debug = config["debug"]
        super(DayMaker, self).__init__(config)

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
        for i in range(rglen-1):
            if i == 0:
                continue
            d = rg[i]
            week_ft = [self.placeholder] * 7 
            week_ft[time_range_weekday[d]] = 1 
            trend = week_ft[:]
            ys = [] # for train
            ys_ = [] # for test
            for j in range(self.stocklen): 
                prev_data = data[j][i-1][0:4]
                one_data = data[j][i][0:4]
                next_data = data[j][i+1][0:4]

                Xi = [prev_data[3]]
                Xi.extend(one_data[0:4])
                yi = yi_ = 0
                if one_data[3] > 0:
                    yi_ = next_data[3] - one_data[3]
                    if yi_ > 0:
                        yi = next_data[1] - one_data[3]
                    elif yi_ < 0:
                        yi = next_data[2] - one_data[3]
                    yi = yi / one_data[3]
                    yi_ = yi_ / one_data[3]
                    if abs(yi) > 0.13:
                        yi = yi_ = 0
                if sum(one_data[0:4]) < 0.1:
                    yi = yi_ = 0
                if yi == 0:
                    Xi = [self.placeholder] * 5

                ys.append(yi)
                ys_.append(yi_)
                Xi = l2Norm(Xi, self.placeholder)
                #Xi = firstNorm(Xi, self.placeholder)
                trend.extend(Xi)
            
            # index sh data
            sh_y_i = index_sh[i+1][3] - index_sh[i][3]
            if d >= self.config["tdates"][0] and d <= self.config['tdates'][1]:
                Xt.append(trend)
                yt.extend(ys_)
                y_sh.append(sh_y_i)
            elif d >= self.config["trdates"][0] and d <= self.config['trdates'][1]:
                X.append(trend)
                y.extend(ys)
        X = np.array(X).astype(np.float32)
        y_ = np.array(y).astype(np.float32)
        Xt = np.array(Xt).astype(np.float32)
        yt_ = np.array(yt).astype(np.float32)
        mean, std = self.getMeans(X, Xt, onelen=5, hlen=7)
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
        ids = map(lambda x: np.random.random(7).tolist(), 
                range(self.stocklen))
        np.save(self.caffe_dir+"/ids.npy", np.array(ids))
        return 
    
def main():
    conf = {
            "debug": True,
            "thread_num": 16,

            # current pro data
            "dates": ['2008-01-01', '2016-05-28'],
            "trdates": ['2013-01-02', '2016-05-28'],
            "tdates": ['2010-01-01', '2013-01-01'],

            ##### test: 2016-05-28 - 2016-08-05
            #"dates": ['2016-05-28', '2016-08-05'],
            #"trdates": ['2016-05-28', '2016-06-05'],
            #"tdates": ['2016-06-06', '2016-08-05'],
            }
    t = DayMaker(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

