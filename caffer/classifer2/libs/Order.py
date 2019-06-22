#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Order.py
# Date: 2016 Mon 10 Oct 2016 10:57:35 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from sms import sendTemplateSMS
from Cache import Cache

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Order(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Order init")
        self.config = config
        self.debug = config["debug"]
        #super(Order, self).__init__(config)
        self.cache = Cache({
            "debug": False, 
            "prefix": "trade_", 
            "host": "localhost"})
        self.orders = self.cache.jsonGet("order")
        if not self.orders:
            self.orders = {}
        print self.orders
        if config.has_key('stop_lose'):
            self.stop_lose = config["stop_lose"]
        else:
            self.stop_lose = 0.03
    
    def symbols(self):
        return self.orders.keys()

    def stats(self):
        print self.orders
        print self.cache.jsonGet("order")
        return

    def _open(self, symbol, price):
        print "open:", symbol, price

    def _close(self, symbol, price):
        print "close:", symbol, price

    # for the stop loss checking
    def tick(self, symbol_prices):
        closed = []
        for pair in symbol_prices:
            symbol, price = pair
            if self.orders.has_key(symbol):
                open_price = self.orders[symbol]["open_price"]
                if open_price <= 0:
                    continue
                profit = price - open_price
                s = 1.0
                if self.orders[symbol]['opt'] == 1:
                    s = 1
                else:
                    s = -1
                profit *= s

                # stop loss
                if profit*1.0/open_price <= -self.stop_lose:
                    self.close(symbol, price)
                    closed.append([symbol, price])

                # change the stop loss by change open
                if profit > 0:
                    self.change(symbol, "open_price", price)
        return closed

    # default up: opt=1
    def open(self, symbol, price, opt=1): 
        if self.orders.has_key(symbol):
            return

        self.orders[symbol] = {
                "open_price": price,
                "close_price": 0,
                "profit_max": 0,
                "opt": opt,
                "status": 0, # 0, new; 1, opened; 2, closing;
                }

        self.orders[symbol]['status'] += 1
        self._open(symbol, price)
        self.cache.jsonSet("order", self.orders)
        return

    def close(self, symbol, price):
        if self.orders[symbol]["status"] == 1:
            self.orders[symbol]["close_price"] = price
            self.orders[symbol]["status"] = 2

            self._close(symbol, price)
            del self.orders[symbol]
            self.cache.jsonSet("order", self.orders)
        return

    def change(self, symbol, k, v):
        self.orders[symbol][k] = v
        self.cache.jsonSet("order", self.orders)
        return

    # TODO
    def fin(self, symbol, price, callback):
        if callback():
            self.orders[symbol]["status"] += 1
            if self.orders[symbol]["status"] > 2:
                del self.orders[symbol]
                print "sold %s" % symbol
            else:
                print "bought %s" % symbol
        else:
            pass

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "stop_lose": 0.03,
            }
    t = Order(conf)
    t.open("000758", 8.15)
    t.open("002085", 19.2)
    t.open("000656", 5.23)
    """
    t.open("001", 100)
    #t.open("002", 88)
    print t.tick([["001", 101]])
    print t.tick([["001", 99]])
    print t.tick([["001", 98]])
    print t.tick([["001", 97]])
    """
    t.stats()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

