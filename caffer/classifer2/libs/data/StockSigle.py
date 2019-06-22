#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: StockSigle.py
# Date: 2016 Mon 01 Aug 2016 09:50:51 AM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import random
import h5py
sys.path.append("/datas/lib/py")
from PyData import PyData

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class StockSigle(PyData):
    ############# rewrite start #############
    def _configure(self):
        self.batch_size = 128
        self.y_threshold = 0.0
        self.tops = [
                (self.batch_size, 1, 1, 1000), 
                (self.batch_size, 20),
                ]
        pass

    def loadStockDatas(self):
        return self.loadStockH5("test.h5")

    def loadStockId(self):
        return ids

    def loadMeanStd(self):
        return mean, std

    def loadX(self, Xs, idx, blen, y, y_):
        s = 5
        w = 7
        l = 800
        l_ = 2862
        X, y = self._loadXFromIdx(Xs, idx, blen, [s, w, l, l_], 
                False, y, y_)
        return X, y

    def batch(self):
        if self.use_seperate_idx > random.random() \
                and self.is_train:
            idx_len = len(self.random_idxs)
            idx_i = random.randint(0, idx_len-1)
            if self.one_epoch_is[idx_i] >= self.rand_lens[idx_i] - self.batch_size:
                self.random_idxs[idx_i] = np.random.permutation(self.idxs[idx_i])
                self.one_epoch_is[idx_i] = 0

            idx = self.random_idxs[idx_i][self.one_epoch_is[idx_i]: self.one_epoch_is[idx_i]+self.batch_size]
        else:
            if self.one_epoch_i >= self.rand_len - self.batch_size:
                self.random_idx = np.random.permutation(self.idx)
                if not self.is_train:
                    self.random_idx = self.random_idx[0:self.rand_len]
                    self.random_idx.sort()
                self.one_epoch_i = 0

            idx = self.random_idx[self.one_epoch_i: self.one_epoch_i+self.batch_size]
        assert idx.shape[0] == self.batch_size, (
                self.one_epoch_i, idx.shape, self.batch_size, 
                self.idx.shape, self.rand_len, 
                )
        self.one_epoch_i += self.batch_size

        y = self.y[idx]
        y_ = self.y_[idx]
        X, y_muti = self.loadX(self.X_base, idx, self.batch_size, y, y_)
        if self.X_noise_per > 0 and self.is_train:
            noise = (np.random.random(X.shape[-1])-0.5) * self.X_noise_per + 1
            X *= noise
        #print y
        res = [X, y_muti, y, y_]
        
        return self.batchTransform(res)

    # if you want to add some new const data
    def batchTransform(self, res):
        return res

    def afterLoadData(self):
        self.idx = np.arange(self.dlen)[self.idx_skip:]
        self.random_idx = np.random.permutation(self.idx)
        self.random_idxs = np.array(map(lambda x: 
                np.random.permutation(x),
                self.loadIdxs(self.y)))
        self.one_epoch_i = 0
        self.one_epoch_is = [0] * len(self.random_idxs)
        self.rand_len = len(self.idx)
        self.rand_lens = map(lambda x: len(x), self.random_idxs)
        self.zeros_init = False
        if not self.is_train:
            self.rand_len = 10240 * self.test_n
            self.random_idx = self.random_idx[0:self.rand_len]
            self.random_idx.sort()
        else:
            self.y_ = self.normY_(self.y_, scale=self.y_log_scale)
            padding_len = len(self.label_padding)
            if padding_len > 0:
                for label in range(padding_len):
                    times = self.label_padding[label]
                    pad_idx = np.array([])
                    if times > 0:
                        y = self.y.reshape(self.y.shape[0])
                        idx = np.argwhere(y[self.idx_skip:] == label)
                        idx.shape = idx.shape[0]
                        for i in range(times):
                            pad_idx = np.concatenate((pad_idx, idx))
                        self.rand_len += pad_idx.shape[0]
                        self.idx = np.concatenate((self.idx, pad_idx)).astype(np.int)
                        self.random_idx = np.random.permutation(self.idx)
                        """
                        print y.shape
                        print idx.shape
                        print idx
                        print pad_idx.shape
                        print self.rand_len
                        print self.batch_size
                        print self.idx.shape
                        print  self.random_idx
                        print self.idx
                        exit()
                        """

    def loadIdxs(self, y_pri):
        y = y_pri.copy()
        y.shape = y.shape[0]
        labels = set([])
        for one in y:
            labels.add(int(one))
        idxs = []
        for l in range(len(labels)):
            idx = np.argwhere(y == l)
            idx.shape = idx.shape[0]
            idxs.append(idx)
        return idxs

    def _loadXFromIdx(self, Xs, idx, blen, swl, mutiy=False, y=[], y_=[], meanstd_index=-1):
        # one len, week len, trend len, stock len
        s, w, l, l_ = swl

        # init the X and y, zeros
        if self.zeros_init == False:
            self.Xzeros = np.zeros((blen, 1, 1, w+s*(1+l)))
            if mutiy:
                self.yzeros = np.zeros((blen, 1, 1, 20))
            self.zeros_init = True

        ii = 0
        t = time.time()
        if mutiy:
            assert len(y_) > 0 and len(y) > 0
            y_res = self.yzeros
        else:
            y_res = y

        X = self.Xzeros
        X_index = 0
        for one_i in idx:
            # make Xi
            i = one_i / l_
            j = one_i % l_
            patten = Xs[i][0][0][w+j*s : w+j*s+s]
            if self.use_id:
                week = self.ids[j]
            else:
                week = Xs[i][0][0][0 : w]
            trend = Xs[i][0][0][w : w+s*l]
            if meanstd_index == -1:
                one_X = self.transform(
                        np.concatenate( (patten, week, trend) ),
                        self.mean, self.std, 255
                        )
            else:
                one_X = self.transform(
                        np.concatenate( (patten, week, trend) ),
                        self.mean[meanstd_index], 
                        self.std[meanstd_index], 255
                        )
            X[X_index][0][0] = one_X

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

        X = X.astype(np.float32)
        y_res = y_res.astype(np.float32)
        return X, y_res

    ############# rewrite end #############

    def loadData(self):
        self.X_base, self.y, self.y_ = self.loadStockDatas()
        self.y = self.redefLabel(self.y, self.y_, self.y_threshold)
        self.mean, self.std = self.loadMeanStd()
        if self.use_id: #TODO
            self.ids = self.loadStockId()
            mean_len = self.mean.shape[0]
            point = (mean_len - 7) / 801
            self.mean[point: point+7] = 0.5
            self.std[point: point+7] = 0.28685906
        self.dlen = len(self.y)
        self.afterLoadData()

    ################### utils
    def loadStockH5(self, fname):
        with h5py.File(fname, 'r') as f:
            X = f['data'][:]
            y = f['label'][:] 
            y_ = f['realy'][:]

        return np.array(X).astype(np.float32), \
                np.array(y).astype(np.float32).reshape(len(y), 1, 1, 1), \
                np.array(y_).astype(np.float32)

    def redefLabel(self, y, y_, threshold=0.0):
        y *= 0.0
        y_.shape = y.shape

        if self.up_threshold != 0:
            xargs = np.argwhere(y_ > self.up_threshold)
        else:
            xargs = np.argwhere(y_ > threshold)
        y[xargs] = 1.0

        if self.sell_threshold != 0:
            xargs = np.argwhere(y_ < self.sell_threshold)
        else:
            xargs = np.argwhere(y_ < -threshold)
        y[xargs] = 2.0

        labels = {}
        for one in y:        
            one = int(one)       
            k = str(one)         
            if labels.has_key(k):
                labels[k] += 1       
            else:
                labels[k] = 1        
        print "labels: ", str(labels)
        return y

    def normY_(self, y_, scale=66):
        # transform
        if scale > 0:
            y_ *= scale
            y_[ np.where(y_ > 0) ] = np.log(y_[ np.where(y_ > 0) ] + 1)
            y_[ np.where(y_ < 0) ] = -np.log(abs(y_[ np.where(y_ < 0) ]) + 1)

        # to 0-1
        y_pos = y_[ np.where(y_ > 0) ]
        max_pos = y_pos.max()
        y_[ np.where(y_ > 0) ] /= max_pos
        y_pos = y_[ np.where(y_ > 0) ]
        print y_pos.max(), y_pos.min()

        y_neg = y_[ np.where(y_ < 0) ]
        min_neg = y_neg.min()
        y_[ np.where(y_ < 0) ] /= abs(min_neg)
        y_neg = y_[ np.where(y_ < 0) ]
        print y_neg.max(), y_neg.min()

        # balance the pos and neg
        if len(self.label_padding) == 0:
            y_pos = y_[ np.where(y_ > 0) ]
            y_neg = y_[ np.where(y_ < 0) ]
            per = abs(y_pos.mean() * 1.0 / y_neg.mean())
            if per < 1:
                y_[ np.where(y_ < 0) ] *= per
            else:
                y_[ np.where(y_ > 0) ] /= per

            per = len(y_pos) * 1.0 / len(y_neg)
            if per < 1:
                y_[ np.where(y_ < 0) ] *= per
            else:
                y_[ np.where(y_ > 0) ] /= per

            y_pos = y_[ np.where(y_ > 0) ]
            y_neg = y_[ np.where(y_ < 0) ]
        print y_pos.max(), y_pos.min()
        print y_neg.max(), y_neg.min()
        print y_pos.mean(), y_neg.mean()

        self.label_mean = y_pos.mean()
        return y_


# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

