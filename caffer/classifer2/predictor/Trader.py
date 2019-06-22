#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Trader.py
# Date: 2016 Sun 09 Oct 2016 09:56:54 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
sys.path.append("/.jingyi/codes/caffer/classifer2/libs")
from lib.Util import *
from Order import Order
from PredictProMerge import PredictProMerge
from sms import sendTemplateSMS
from lib.StockTs import StockTs

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Trader(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Trader init")
        self.config = config
        self.debug = config["debug"]
        #super(Trader, self).__init__(config)
        self.order = Order(config)
        self.predictor = PredictProMerge(config)
        self.ts = StockTs(config)
    
    def _configure(self):
        return

    def configure(self):
        self._configure()
        return

    def schedule(self):
        dt_now = mkDate()
        wkday = weekday(dt_now)
        if wkday > 4:
            return

        tm = mktime('%H:%M:%S')
        if (tm >= "10:00:00" and tm <= "11:30:00") \
                or (tm >= "13:00:00" and tm < "14:30:00"): # ticking
                    self.tickTrace()
        elif tm == "14:45:00": # sell all
            self.closeAll()
        elif tm == "14:50:00": # buying
            self.openAll()
        elif tm == "03:00:00": # data download time
            self.cache()
        elif True: # test
            """
            self.openAll()
            exit()
            self.tickTrace()
            time.sleep(1)
            time.sleep(20)
            self.closeAll()
            self.cache()
            """
            pass
        else:
            pass
        return

    def mainLoop(self):
        while True:
            self.schedule()
            time.sleep(1)
        return

    def predict(self):
        dt_now = mkDate()
        #dt_now = "2016-09-01"
        dt_past, dt_now = mkPrevDate(dt_now, 80)

        self.predictor.now = dt_now
        res = self.predictor.doPredict(
                sys.argv[1], # net
                sys.argv[2], # model
                dt_past,
                dt_now,
                "./caffe_merge",
                )
        return res

    def openAll(self):
        predicts = self.predict()
        data = []
        for one in predicts[0:5]:
            tmp = str(one[0]) + '_' + str(one[1])
            data.append(tmp)
            self.order.open(one[0], one[1])
        sendTemplateSMS("13811631314", data, 121534)
        print self.order.orders
        return

    def tickTrace(self):
        symbols = self.order.symbols()
        if len(symbols) > 0:
            info = self.ts.crrDataSymbol(symbols)
            data = map(lambda x: [x[0], float(x[1])], info)
            res = self.order.tick(data)
            if len(res) > 0:
                res = map(lambda x: str(x[0])+"_"+str(x[1]), res)
                sendTemplateSMS("13811631314", res, 121534)
            print "tick:", res
        return

    def closeAll(self):
        symbols = self.order.symbols()
        for one in symbols:
            self.order.close(one, 0)
        print "close:", symbols
        sendTemplateSMS("13811631314", ["11111"], 48395)
        print self.order.orders
        return

    def cache(self):
        self.predict()
        return

    def run(self):
        self.configure()
        self.mainLoop()
        return

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,

            "gpu": 0,
            "thread_num": 18,

            "stop_lose": 0.02,
            }
    t = Trader(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

