#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: PredictProMerge.py
# Date: 2016 Tue 04 Oct 2016 08:36:08 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from PredictTestMerge import PredictTestMerge
from lib.Util import *
from lib.MyMath import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class PredictProMerge(PredictTestMerge):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: PredictProMerge init")
        self.config = config
        self.debug = config["debug"]
        super(PredictProMerge, self).__init__(config)

    def run(self):
        if len(sys.argv) == 4:
            dt_now = sys.argv[3]
            assert len(dt_now.split("-")) == 3
        else:
            dt_now = mkDate()

        dt_past, dt_now = mkPrevDate(dt_now, 80)
        self.now = dt_now

        self.doPredict(
                sys.argv[1], # net
                sys.argv[2], # model
                dt_past,
                dt_now,
                "./caffe_merge",
                )
        return

    def dataAppend(self, data, index_sh, rg, rg_wkday):
        if rg[-1] != self.now:
            print rg[-1], self.now
            while True:
                try:
                    crr_data = self.dmaker_rt.crrData()
                    break
                except:
                    print "network err, retrying..."
                    time.sleep(0.3)
                    continue
            data = np.array(data)
            print data.shape
            print len(crr_data)
            todays = np.zeros((data.shape[0], 1, 6))
            for one in crr_data:
                sid = self.dmaker_rt.stockID(one[0], True)
                index = int(sid)
                if sid != -1 and index < self.stocklen:
                    tmp = np.array([ 
                            one[1], one[2],
                            one[3], one[4],
                            one[5], 0, 
                            ])
                    if tmp[0] * tmp[1] * tmp[2] * tmp[3] == 0:
                        tmp = tmp * 0.0

                    # always do padding
                    todays[index][0] = tmp

            calc = data[:,-1,:].reshape(todays.shape)
            print calc.shape
            print (calc[:, 0, 0:4] - todays[:, 0, 0:4]).sum()
            print ((calc[:, 0, 0:4] - todays[:, 0, 0:4]) * todays[:, 0, 0:4]).sum()
            if ((calc[:, 0, 0:4] - todays[:, 0, 0:4]) * todays[:, 0, 0:4]).sum() != 0:
                #print rg_wkday, rg
                #index_sh.append(0)
                rg_wkday[self.now] = weekday(self.now)
                rg = np.concatenate((rg, np.array([self.now])))
                #print rg_wkday, rg
                data = np.concatenate((data, todays), axis=1)
                print "today padded to data"
            print np.array(data).shape
        return data, index_sh, rg, rg_wkday

    def predictLoop(self, data, index_sh, rg, rg_wkday):
        rglen = len(rg)
        prevs = np.zeros((self.stocklen, 3, 5)) 
        prev_y = np.zeros(self.stocklen)

        d = rg[-1]
        week_ft = [self.placeholder] * 7 
        week_ft[weekday(d)] = 1 
        trend_price = week_ft[:]
        trend_rsi = week_ft[:]
        trend_ave = week_ft[:]
        prices = []
        for sid in range(self.stocklen):
            # data calc
            pprice, prsi, pma = self.calcOneDay(data, sid, rglen-2)
            price, rsi, ma = self.calcOneDay(data, sid, rglen-1)
            prices.extend(price)

            # calc dx
            dprice = np.array(price) - np.array(pprice)
            drsi = np.array(rsi) - np.array(prsi)
            dma = np.array(ma) - np.array(pma)
            dprice = l2Norm(dprice, self.placeholder)
            drsi = l2Norm(drsi, self.placeholder)
            dma = l2Norm(dma, self.placeholder)
            trend_price.extend(dprice)
            trend_rsi.extend(drsi)
            trend_ave.extend(dma)

            # calc x
            price = l2Norm(price, self.placeholder)
            rsi = l2Norm(rsi, self.placeholder)
            ma = l2Norm(ma, self.placeholder)
            trend_price.extend(price)
            trend_rsi.extend(rsi)
            trend_ave.extend(ma)

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
        res = self.resShow(y, self.dmaker_rt.stockCode, 20)
        prices = np.array(prices)
        prices.shape = (2862, 5)
        code_price = []
        for one in res:
            print one
            print prices[one[2]]
            code_price.append([one[1], round(prices[one[2]][-1], 2)])
        print y.shape
        print code_price
        return code_price

    # j: stock_id
    # i: the time index
    def calcOneDay(self, data, j, i):
        prev_data = data[j][i-1][0:4]
        one_data = data[j][i][0:4]

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
        return price, rsi, ma

def main():
    conf = {
            "debug": True,
            "gpu": 0,
            "thread_num": 15
            }
    t = PredictProMerge(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

