#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DMaker.py
# Date: 2016 Sun 07 Aug 2016 07:46:21 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import h5py, random
sys.path.append("/datas/lib/py")
from lib.StockTs import StockTs
from lib.MyStock import MyStock

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DMaker(MyStock):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DMaker init")
        self.config = config
        self.debug = config["debug"]
        self.ts = StockTs(config)
        config["identifier"] = 'test'
        super(DMaker, self).__init__(config)
        self.caffe_dir = "caffe/"
        self.h5iter = 0
        self.stocklen = 2862 # magic number
    
    def _oneDay(self, dt, stock_code, is_index=False):
        return self.ts.dayData(stock_code, dt, is_index)

    def _rangeDays(self, code, from_dt, to_dt,
            rgkeys, is_index=False):
        return self.ts.rangeData(code, from_dt,
                to_dt, rgkeys, is_index)

    def _baseCodes(self, fname):
        ls = self.ts.baseCodes()
        ls = sorted(map(lambda x: int(self.stockID(x, True)), ls))
        return ls

    def __del__(self):
        pass

    def getMeans(self, X, Xt, onelen=24, hlen=0):
        #X = np.concatenate((X, Xt))
        print X.shape
        header = X[:, 0:hlen]
        X = X[:, hlen:]
        print X.shape, onelen, hlen

        assert X.shape[0] * X.shape[1] % onelen == 0
        n = X.shape[0] * X.shape[1] / onelen
        X_reshape = X.reshape(n, onelen)
        mean_one = X_reshape.mean(0)
        std_one = X_reshape.std(0)

        mean = X.mean(0)
        std = X.std(0)
        if hlen > 0:
            mean_h = header.mean(0)
            std_h = header.std(0)
            mean = np.concatenate((mean_h, mean))
            std = np.concatenate((std_h, std))
        mean = np.concatenate((mean_one, mean))
        std = np.concatenate((std_one, std))
        return mean, std

    def counter(self, y):
        counter = {}
        for one in y:
            if counter.has_key(one):
                counter[one] += 1
            else:
                counter[one] = 1
        return counter

    def labelMaker(self, y):
        if y == 0:
            return 0
        else:
            return 1 if y > 0 else 2

    def saveH5DB(self, X, Xf, y, y_, rand=True, prefix="data"):
        if rand:
            X, Xf, y, y_ = self.randXY(X, Xf, y, y_)
        X = np.array(X).astype(np.float32)
        Xf = np.array(Xf).astype(np.float32)
        y = np.array(y).astype(np.float32)
        y_ = np.array(y_).astype(np.float32)
        print X, X.sum()
        print "data shape", X.shape
        print "y, y_: max, min, std, mean"
        print y.max(), y.min(), y.std(), y.mean()
        print y_.max(), y_.min(), y_.std(), y_.mean()
        dirname = os.path.abspath(self.caffe_dir)
        fname = os.path.join(dirname, "%s_%d.h5" % (prefix, self.h5iter))
        with h5py.File(fname, 'w') as f:
            if len(X.shape) == 2:
                f['data'] = X.reshape((len(X), 1, 1, len(X[0])))
                f['dataf'] = Xf.reshape((len(Xf), 1, 1, len(Xf[0])))
            else:
                f['data'] = X
                f['dataf'] = Xf
            f['label'] = y
            f['realy'] = y_
        self.h5iter += 1
        return fname

    def randXY(self, X, Xf, y, y_):
        ls = []
        for i in range(len(X)):
            ls.append([X[i], Xf[i], y[i], y_[i]])
        random.shuffle(ls)
        r_X = []
        r_Xf = []
        r_y = []
        r_y_ = []
        for one in ls:
            r_X.append(one[0])
            r_Xf.append(one[1])
            r_y.append(one[2])
            r_y_.append(one[3])
        return r_X, r_Xf, r_y, r_y_

    def periodInfo(self, arr):
        assert len(arr) > 0
        assert len(arr[0]) >= 5

        len_arr = len(arr)
        res = {
                "len": 0,

                "walk_sum": 0,
                "walk_sum_abs": 0,
                "up_walk": 0,
                "sell_walk": 0,
                "ave_close": 0,
                "volume": 0, # TODO

                "rsi": 0,
                }

        last_close = 0
        for i in range(len_arr):
            one = arr[i]
            if sum(one[0:4]) < 0.1:
                continue
            if last_close == 0:
                last_close = one[0]
            close = one[3]
            walk = close - last_close
            walk_abs = abs(walk)
            res['walk_sum'] += walk
            res['walk_sum_abs'] += walk_abs
            if walk > 0:
                res['up_walk'] += walk
            elif walk < 0:
                res['sell_walk'] += walk

            res['ave_close'] = 1.0 * (close - res['ave_close']) \
                    / (res['len'] + 1) + res['ave_close']
            last_close = close
            res['len'] += 1
        if res['up_walk'] + abs(res['sell_walk']) <= 0:
            res['rsi'] = 0.5
        else:
            res['rsi'] = res['up_walk'] * 1.0 \
                / (res['up_walk'] + abs(res['sell_walk']))
        res['rsi'] += 0.5
        return res

def main():
    conf = {
            "debug": False,
            }
    t = DMaker(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

