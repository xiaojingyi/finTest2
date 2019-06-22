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
from Base import Base
from DataFactory import DataFactory

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
        super(PredictorBase, self).__init__(config)

    def setTData(self):
        self.data_factory = DataFactory(self.config)
        self.data_handler = self.data_factory.init(self.config['data_type'])
        lx, lt = self.data_handler.placeAllData(*self.mkData())
        print "train len: %d, test len: %d" % (lx, lt)
        return
    
    def initNet(self, net, model):
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        self.netname = net
        self.model = model
        self.iter = 0
        return

    def predict(self, X, use_train_net=False, usesig=False):
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
            res = self.testNetRes(X, usesig)

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

    def testNetRes(self, X, usesig=False):
        #print "using test net..."
        Xlen = X.shape[0]
        X = self.padX(X)
        #print Xlen, X.shape
        X = X.astype(np.float32)
        y = np.zeros((X.shape[0], 1, 1, 1)) \
                .astype(np.float32)

        self.net.set_input_arrays(X, y)
        res = None
        index = "out_1"
        if usesig:
            index = "sig_1"
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

        #print res.shape
        if not usesig:
            res = np.array(map(lambda x: self.softmax(x), res))
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

