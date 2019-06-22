#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: ForexSingle.py
# Date: 2016 Sat 17 Sep 2016 05:42:23 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
from StockSigle import StockSigle
from lib.threadpool import ThreadPool

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class ForexSingle(StockSigle):
    ############# rewrite start #############
    def _configure(self):
        pass

    def loadStockDatas(self):
        pass

    def loadMeanStd(self):
        pass

    ############# public start #############
    def Xthread(self, Xs, X, X_index, one_i, atom_num):
        one_X = None
        for i in range(atom_num):
            tmp = (Xs[one_i][0][0] - Xs[one_i-i-1][0][0]) / (i+1)
            if one_X == None:
                one_X = tmp
            else:
                one_X = np.concatenate((one_X,tmp))

        X[X_index][0][0] = self.transform(
                one_X, self.mean, self.std, 255
                )

        return 

    def _loadXFromIdx(self, Xs, idx, blen, atom_num, mutiy=False, y=[], y_=[]):
        # init the X and y, zeros
        atom_len = Xs.shape[-1]
        if self.zeros_init == False:
            feat_len = atom_num * atom_len
            self.Xzeros = np.zeros((blen, 1, 1, feat_len))
            if mutiy:
                self.yzeros = np.zeros((blen, 1, 1, 20))
            self.zeros_init = True

        t = time.time()
        if mutiy:
            assert len(y_) > 0 and len(y) > 0
            y_res = self.yzeros
        else:
            y_res = y

        X = self.Xzeros
        X_index = 0
        #pool = ThreadPool(2)
        one_X = np.zeros(X.shape[-1])
        for one_i in idx:
            """
            pool.add_task(self.Xthread, *[Xs, X, X_index, one_i, atom_num])
            one_X = None
                if one_X == None:
                    one_X = tmp
                else:
                    one_X = np.concatenate((one_X,tmp))
            """
            for i in range(atom_num):
                one_X[i*atom_len:(i+1)*atom_len] = \
                        (
                                Xs[one_i][0][0] 
                                - Xs[one_i-i*5-1][0][0]
                                ) / (i*5+1)

            X[X_index][0][0] = self.transform(
                    one_X, self.mean, self.std, 255
                    )

            # make muti-dim label
            # NOTE: must have self.label_mean(called normY_)!!!
            if mutiy:
                y_res[X_index] *= 0
                tmp = abs(y_[X_index]).reshape(1)
                if y[X_index] == 0:
                    y_res[X_index][0][0][0] = self.label_mean
                else:
                    y_res[X_index][0][0][int(y[X_index])] = tmp
            X_index += 1
        #pool.destroy()
        X = X.astype(np.float32)
        y_res = y_res.astype(np.float32)
        return X, y_res

    def loadX(self, Xs, idx, blen, y, y_):
        atom_num = 60
        X, y = self._loadXFromIdx(Xs, idx, blen, atom_num, True, y, y_)
        return X, y

def main():
    conf = {
            "debug": True,
            }
    t = ForexSingle(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

