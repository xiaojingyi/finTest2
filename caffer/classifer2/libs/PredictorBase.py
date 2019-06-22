#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: PredictorBase.py
# Date: 2016 Sat 16 Jul 2016 09:19:02 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import caffe
import numpy as np
sys.path.append("/datas/lib/py")
from lib.MyMath import *
from lib.Util import *
from Base import Base

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class PredictorBase(Base):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: PredictorBase init")
        self.config = config
        self.debug = config["debug"]
        if self.config.has_key("gpu"):
            self.gpu_id = self.config['gpu']
        else:
            self.gpu_id = 0
        self.stocklen = 2862
        super(PredictorBase, self).__init__(config)

    # y_predict: show be the ordered
    # choose_len: default we just choose the 300 + 500
    def resShow(self, y_predict, id2codeCallback, 
            nlimit=20, choose_len=800):
        res = map(lambda x: (
            x[1][np.argmax(x[1])],
            x[0],
            np.argmax(x[1]),
            ), enumerate(y_predict))
        res = sorted(res[0:choose_len], key=lambda x: x[0], reverse=True)
        ct = 0
        stock_arr = []
        for i in range(choose_len):
            one = res[i]
            if one[2] == 1:
                stock_id = one[1]
                stock_code = id2codeCallback(stock_id)
                if stock_code[0:3] != '300': # ignore chuangyeban
                    if ct < nlimit:
                        data_one = [i, stock_code, stock_id]
                        stock_arr.append(data_one)
                    ct += 1
        return stock_arr

    def toCaffeX(self, trend, onelen=5, prices=[]):
        trend = np.array(trend)
        X = np.zeros((self.stocklen, 1, 1, onelen * 801 + 7))
        for i in range(self.stocklen):
            Xi = trend[7+i*onelen: 7+(i+1)*onelen]
            if len(prices) > 0:
                tmp = prices[i*5: (i+1)*5]
                walk = abs(tmp[4]-tmp[0])/(tmp[0]+self.placeholder)
                if walk > 0.098:
                    Xi = np.zeros(onelen) + self.placeholder
                    Xi = l2Norm(Xi, self.placeholder)
            X[i][0][0] = np.concatenate((Xi, trend[0: 7+800*onelen]))
        return X

    def initNet(self, net, model):
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        self.netname = net
        self.model = model
        self.iter = 0
        return

    def predictMutiModel(self, X, models=[], use_train_net=False, index="sigmoid_1"):
        res = np.zeros(1)
        for one in models:
            self.model = one
            tmp = self.predict(X, use_train_net, index)
            if res.sum() == 0:
                res = tmp
            else:
                res *= tmp
        return res

    def predict(self, X, use_train_net=False, index="sigmoid_1"):
        #print "used:", index
        if use_train_net:
            self.net = caffe.Net(self.netname, str(self.model), caffe.TRAIN)
        else:
            self.net = caffe.Net(self.netname, str(self.model), caffe.TEST)
        self.batch_shape = self.net.blobs['data'].data.shape
        assert len(X.shape) == len(self.batch_shape)

        if use_train_net:
            #TODO
            print "todo"
            exit()
            res = self.trainNetRes(X)
        else:
            res = self.testNetRes(X, index)

        del self.net
        return res

    # TODO, not better than testnet
    def trainNetRes(self, X):
        print "using train net..."
        i = 0
        res = []
        for i in range(X.shape[0]):
            Xi = X[i]
            Xt = np.zeros(self.batch_shape) + Xi
            Xt = Xt.astype(np.float32)
            yt = np.zeros((self.batch_shape[0], 1, 1, 1)) \
                    .astype(np.float32)
            self.net.set_input_arrays(Xt, yt)
            self.net.forward()
            out = self.net.blobs["out_1"].data
            ave = out.mean(0)
            ave = self.softmax(ave)
            res.append(ave)

            if i>0 and i*self.batch_shape[0] % (128*2000) == 0:
                del self.net
                self.net = caffe.Net(self.netname, str(self.model), caffe.TRAIN)
                print i

        return np.array(res)

    def testNetRes(self, X, index = "out_1"):
        #print "using test net..."
        Xlen = X.shape[0]
        X = self.padX(X)
        #print Xlen, X.shape
        X = X.astype(np.float32)
        y = np.zeros((X.shape[0], 1, 1, 1)) \
                .astype(np.float32)

        self.net.set_input_arrays(X, y)
        res = None
        
        if X.shape[0] > self.batch_shape[0]:
            num = X.shape[0] / self.batch_shape[0]
            for i in range(num):
                self.net.forward()
                if res != None:
                    res = np.concatenate((
                        res, 
                        self.net.blobs[index].data,
                        ))
                else:
                    res = self.net.blobs[index].data
        else:
            self.net.forward()
            res = self.net.blobs[index].data

        return res[0:Xlen]

    # merge the out info into index info
    def outCalc(self, out, alpha=0.5):
        res = []
        for one in out:
            l = np.argmax(one)
            res.append([one[l], int(l)])

        res = sorted(res, key=lambda x: x[0], reverse=True)
        l_info = [0, 0, 0]
        for i in range(int(len(res) * alpha)):
            one = res[i]
            score = one[0]
            label = one[1]
            if label == 0:
                continue
            l_info[label] += score

        return np.argmax(l_info)

    def padX(self, X):
        if X.shape[0] % self.batch_shape[0] != 0:
            tmp = np.zeros((
                self.batch_shape[0]-X.shape[0]%self.batch_shape[0],
                self.batch_shape[1],
                self.batch_shape[2],
                self.batch_shape[3],
                ))
            X = np.concatenate( (X, tmp) )
        return X

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            }
    t = PredictorBase(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

