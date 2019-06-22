#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DayMakerAERLMulti.py
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

class DayMakerAERLMulti(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DayMakerAERLMulti init")
        self.config = config
        self.debug = config["debug"]
        super(DayMakerAERLMulti, self).__init__(config)
    
    def element(self, one, last, _start):
        _open, _high, _low, _close = one[0:4]
        _high = _high if _high>last[2] else last[2]
        last[3] = _low if last[3]<=0 else last[3]
        _low = _low if _low>0 and _low<last[3] else last[3]
        _open =  last[1] if last[1] > 0 else _open
        _close =  _close if _close > 0 else last[4]

        if _start == 0:
            _start = _open
        if _start == 0:
            _start = _close

        tmp = [_start, _open, _high, _low, _close]
        if _start == 0:
            tmp = [0] * 5
        return tmp

    def elementsInfo(self, rd, start_p):
        walk = 0
        _start = start_p
        tmp = [0] * 5
        for one in rd:
            tmp = self.element(one[0:4], tmp, _start)
        walk = tmp[-1] - tmp[0]
        if tmp[0] == 0 or tmp[-1] == 0:
            walk = 0
        else:
            walk = walk * 1.0 / tmp[0]
        return tmp, walk

    def trendMaker(self, data, i, dlen=1):
        week_ft = [self.placeholder] * 7 
        # till now, week_ft ignored
        trend = week_ft[:]
        walks = []
        for j in range(self.stocklen): 
            one_stock = data[j]
            start_p = one_stock[i-1][3]
            crr_rg = one_stock[i:i+dlen]
            tmp, walk = self.elementsInfo(crr_rg, start_p)
            #walk /= dlen
            y_thres = 0.13 * dlen
            if abs(walk) > y_thres:
                walk = 0

            Xi = tmp[:]
            if walk == 0:
                Xi = [self.placeholder] * 5

            Xi = firstNorm(Xi)
            Xi /= dlen
            walk /= dlen
            #print tmp, walk
            #print Xi
            trend.extend(Xi)
            walks.append(walk)
        return trend, walks

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
        for i in range(rglen-22):
            if i < 1:
                continue
            d = rg[i]
            print "making", d
            # index sh data
            sh_y_i = index_sh[i+1][3] - index_sh[i][3]
            if d >= self.config["tdates"][0] and d <= self.config['tdates'][1]:
                dlen = 1
                ft, walk = self.trendMaker(data, i, dlen)
                ftf, walkf = self.trendMaker(data, i+dlen, dlen)
                Xt.append(ft)
                Xft.append(ftf)
                yt.append(walkf)
                yt_.append(walk)
            elif d >= self.config["trdates"][0] and d <= self.config['trdates'][1]:
                for j in range(5):
                    dlen = j+1
                    #dlen = 10
                    ft, walk = self.trendMaker(data, i, dlen)
                    #print ft, walk
                    #exit()
                    ftf, walkf = self.trendMaker(data, i+dlen, dlen)
            
                    X.append(ft)
                    Xf.append(ftf)
                    y.append(walkf)
                    y_.append(walk)
                    y_sh.append(sh_y_i)
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
    
    def daysData(self, di, ti, dlen=10):
        data = []
        walks = []
        dataf = []
        walksf = []
        last = [0] * 5
        lastf = [0] * 5
        _start = di[ti-1][0:4][3]
        _startf = di[ti+dlen-1][0:4][3]
        for i in range(dlen):
            i_crr = ti + i
            tmp = self.element(di[i_crr], last, _start)
            last = tmp[:]
            walk = tmp[-1] - tmp[0]
            if tmp[0] == 0 or tmp[-1] == 0:
                walk = 0
            else:
                walk = walk * 1.0 / tmp[0]

            i_next = ti + dlen + i
            tmpf = self.element(di[i_next], lastf, _startf)
            lastf = tmpf[:]
            walkf = tmpf[-1] - tmpf[0]
            if tmpf[0] == 0 or tmpf[-1] == 0:
                walkf = 0
            else:
                walkf = walkf * 1.0 / tmpf[0]

            data.append(firstNorm(tmp))
            dataf.append(firstNorm(tmpf))
            walks.append(walk)
            walksf.append(walkf)

        return data, walks, dataf, walksf

def main():
    conf = {
            "debug": True,
            "thread_num": 26,

            # current pro data
            "dates": ['2008-01-01', '2017-05-08'],
            "trdates": ['2013-01-02', '2017-05-08'],
            "tdates": ['2010-01-01', '2013-01-01'],

            ##### test: 2008-01-01 - 2016-08-19
            #"dates": ['2008-01-01', '2016-08-19'],
            #"trdates": ['2013-01-02', '2016-08-19'],
            #"tdates": ['2010-01-01', '2013-01-01'],
            }
    t = DayMakerAERLMulti(conf)
    t.run()
    return

if __name__ == "__main__":
    main()


# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

