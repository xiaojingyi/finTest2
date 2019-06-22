#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: TestCounter.py
# Date: 2016 Tue 26 Jul 2016 06:41:12 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import math
sys.path.append("/datas/lib/py")
from lib.Util import *
from Base import Base

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class TestCounter(Base):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: TestCounter init")
        self.config = config

        if not config.has_key("fname"):
            self.config['fname'] = "test_res.csv"
        if not config.has_key("crr_len"):
            self.config['crr_len'] = 20

        self.debug = config["debug"]
        super(TestCounter, self).__init__(config)

        self.res_fname = self.config['fname']
        self.crr_len = self.config['crr_len']
        self.stop_lose = self.config['stop_lose']
        self.init()
    
    def init(self):
        self.label_counter = {}
        self.res = {
                "good": 0, 
                "crr_good": 0,
                "total": 0, 
                "profit": 0,
                "potential": 0,
                }
        self.acc_crr = 0
        self.csv_data = []
        self.crr_acc_info = []
        self.label_counter = {
                '0': 0,
                '1': 0,
                '2': 0,
                }
        return

    def getStatus(self):
        profit_arr = map(lambda x: x[2], self.csv_data)
        if len(profit_arr) > 0:
            a = profit_arr[-1] / (len(profit_arr) + 0.000001)
            std = sum(map(
                lambda i: (profit_arr[i] - a * (i+1)) ** 2, 
                range(len(profit_arr))
                )) / (len(profit_arr) + 0.000001)
            std = math.sqrt(std)
            # acc /= std
            #tmp = self.crr_acc_info[0] / (std + 0.000001)
            # profit_per /= std
            tmp = self.crr_acc_info[3] / (std + 0.000001)
            if a <= 0:
                tmp = 0
        else:
            tmp = 0
        return self.label_counter, self.crr_acc_info + [tmp]

    def addResArr(self, data, one_calc=False):
        for one in data:
            self.addRes(one[0], one[1], not one_calc)
        if one_calc:
            self.calc(True)
        print self.label_counter
        return

    def addRes(self, label, real, is_calc=True):
        # label counter
        l = str(label)
        if self.label_counter.has_key(l):
            self.label_counter[l] += 1
        else:
            self.label_counter[l] = 1

        # the statistic info
        profit = 0
        if label == 1:
            profit = real
        elif label == 2:
            profit = -real

        if label == 1 or label == 2:
            if profit < -self.stop_lose:
                profit = -self.stop_lose
            #if is_calc:
            #    print round(profit, 3), 
            self.res['profit'] += profit
            self.res['potential'] += abs(real)
            self.res['total'] += 1

            if profit > 0:
                self.res['good'] += 1
                crr_good = 1
            else:
                crr_good = 0

            alpha = 2.0 / (self.crr_len + 1)
            self.acc_crr = alpha*crr_good + (1-alpha)*self.acc_crr

        if is_calc:
            self.calc()
        return

    def calc(self, isshow=False):
        if self.res['total'] <= 0:
            return

        acc = self.res['good'] * 1.0 / self.res['total']
        profit = self.res['profit']
        potential = self.res['potential']
        if potential > 0:
            profit_per = profit * 1.0 / potential
        else:
            profit_per = 0
        acc_crr = self.acc_crr

        tmp = [
                acc, acc_crr, profit, 
                profit_per, potential
                ]
        if isshow:
            print round(tmp[0], 3), tmp[2]
        self.crr_acc_info = tmp[:]
        self.csv_data.append(tmp)
        return

    def saveRes(self):
        if len(self.csv_data) > 0:
            mkCsvFileSimple(self.res_fname, self.csv_data)
        return

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = TestCounter(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

