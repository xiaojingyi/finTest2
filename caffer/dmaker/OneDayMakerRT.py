#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: OneDayMakerRT.py
# Date: 2016 Mon 22 Aug 2016 11:09:03 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from OneDayMaker import OneDayMaker
from lib.Util import *
from lib.MyMath import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class OneDayMakerRT(OneDayMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: OneDayMakerRT init")
        self.config = config
        self.debug = config["debug"]
        super(OneDayMakerRT, self).__init__(config)
    
    def crrData(self):
        t = time.time()
        #crr_data = self.ts.crrData("tmp.json")
        crr_data = self.ts.crrData()
        t_ = time.time()
        print "download time", (t_-t)
        return crr_data

    def makeData(self, dt, nostop=False):
        # get the currents
        crr_data = self.crrData()
        # calc the currents
        res = np.zeros((self.stocklen, 5))
        i = 0
        for one in crr_data:
            sid = self.stockID(one[0], True)
            index = int(sid)
            if sid != -1 and index < self.stocklen:
                tmp = np.array([
                        one[5], 
                        one[1], 
                        one[2], 
                        one[3], 
                        one[4], 
                        ])
                if tmp[0] * tmp[1] * tmp[2] * tmp[3] * tmp[4] == 0:
                    tmp *= 0
                res[index] = tmp
            else:
                #print i, one[0], "passed"
                i += 1
        week_ft = [self.placeholder] * 7
        week_ft[weekday(dt)] = 1
        week_ft = np.array(week_ft)
        prices = res.copy()
        trend = res
        for i in range(res.shape[0]):
            if res[i].sum() == 0:
                prices[i] += self.placeholder
                trend[i] += self.placeholder

            trend[i] = l2Norm(trend[i], self.placeholder)
        prices.shape = self.stocklen * 5
        trend.shape = self.stocklen * 5
        trend = np.concatenate((week_ft, trend))
        #print trend.shape, prices.shape
        #exit()
        return trend, prices

    def testPrint(self):
        self.makeData()
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = OneDayMakerRT(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

