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
        return

    def addResArr(self, data, all_in_one=False):
        for one in data:
            self.addRes(one[0], one[1], not all_in_one)

        if all_in_one:
            self.calc()
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

    def calc(self):
        if self.res['total'] <= 0:
            return

        acc = self.res['good'] * 1.0 / self.res['total']
        profit = self.res['profit']
        potential = self.res['potential']
        profit_per = profit * 1.0 / potential
        acc_crr = self.acc_crr

        tmp = [
                acc, acc_crr, profit, 
                profit_per, potential
                ]
        print round(tmp[0], 3), tmp[2]
        self.csv_data.append(tmp)
        return

    def saveRes(self):
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

