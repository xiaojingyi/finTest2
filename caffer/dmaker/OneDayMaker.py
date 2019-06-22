#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: OneDayMaker.py
# Date: 2016 Mon 08 Aug 2016 01:13:41 AM CST
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

class OneDayMaker(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: OneDayMaker init")
        self.config = config
        self.debug = config["debug"]
        super(OneDayMaker, self).__init__(config)
    
    def downData(self, dt):
        prev_day, crr_day = mkPrevDate(dt, 7)
        data, index_sh = self.rangeDaysAll(
                prev_day,
                crr_day,
                )
        return data, index_sh

    def makeData(self, dt, nostop=False):
        data, index_sh = self.downData(dt)
        week_ft = [self.placeholder] * 7
        week_ft[weekday(dt)] = 1
        print week_ft
        trend = week_ft[:]
        prices = []
        print len(data), len(data[0])
        for i in range(self.stocklen):
            crr = data[i][-1][0:4]
            prev = data[i][-2][0:4]

            Xi = [prev[3]]
            Xi.extend(crr)
            if nostop and np.array(crr).mean() == crr[0]:
                Xi = [self.placeholder] * 5
            if sum(crr[0:4]) < 0.1:
                Xi = [self.placeholder] * 5

            prices.extend(Xi)
            Xi = l2Norm(Xi, self.placeholder)
            trend.extend(Xi)
        print len(trend)
        return trend, prices

    def caffeX(self, trend, prices, skip_stops=True):
        trend = np.array(trend)
        res = np.zeros((self.stocklen, 1, 1, 4012))
        base_trend = trend[0:7+5*800]
        for i in range(self.stocklen):
            one = trend[7+i*5: 7+(i+1)*5]
            price = prices[i*5: (i+1)*5]
            crr_walk = abs(price[4] - price[0]) / \
                    (abs(price[0]) + self.placeholder)
            #print crr_walk
            if crr_walk > 0.098 and skip_stops:
                one = one * 0 + self.placeholder
                one = l2Norm(one, self.placeholder)
            Xi = np.concatenate((one, base_trend))
            res[i][0][0] = Xi
        return res

    def getDay(self, dt, save=False, skip_stops=True):
        t = time.time()
        trend, prices = self.makeData(dt)
        res = self.caffeX(trend, prices, skip_stops)
        print res
        if save:
            fname = self.caffe_dir+"/crr_data.npy"
            np.save(fname, res)

        t_ = time.time()
        print "execute time: %.3f" % (t_ - t)
        return res

    def checkDate(self, dt):
        arr = dt.split("-")
        assert len(arr) == 3
        arr = map(lambda x: int(x), arr)
        for one in arr:
            assert one > 0
        return

    def run(self):
        dt = sys.argv[1]
        self.checkDate(dt)
        self.getDay(dt)
        return

def main():
    conf = {
            "debug": True,
            "thread_num": 19,
            }
    t = OneDayMaker(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

