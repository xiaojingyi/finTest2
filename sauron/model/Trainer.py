#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-5-4 22:27:47$
# Note: This source file is NOT a freeware
# Version: Trainer.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-5-4 22:27:47$"

import os, sys, time
import numpy as np
import caffe
import google.protobuf as pb2

class Trainer(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Trainer init")
        self.config = config
        self.debug = config["debug"]
        if not config["close_gpu"]:
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.data_dir = config["data_dir"]
        if not self.data_dir:
            data_dir = "./datas"
        self.init(config["solver"], config["pre_model"], config["solverstate"])
        #super(Trainer, self).__init__(config)
        
    def init(self, solver_prototxt, pretrained_model, solverstate):
        self.model_dir = os.path.dirname(solver_prototxt)
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model:
            self.solver.net.copy_from(pretrained_model)
        if solverstate:
            self.solver.restore(solverstate);
        self.solver_param = caffe.proto.caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as fd:
            pb2.text_format.Merge(fd.read(), self.solver_param)
        self.train_batch_size = self.solver.net.blobs['data'].data.shape[0]
        self.test_batch_size = self.solver.test_nets[0].blobs['data'].data.shape[0]

        return

    def prepare_data(self, train_size, test_size):
        x, y = self.loadData(train_size, self.loadDataTrain)
        print x, y
        print x.shape, y.shape
        self.solver.net.set_input_arrays(x, y)
        x_, y_ = self.loadData(test_size, self.loadDataTest)
        print x_, y_
        print x_.shape, y_.shape
        self.solver.test_nets[0].set_input_arrays(x_, y_)
        
        return

    def train(self, loss=["loss"]):
        t1 = time.time()
        self.solver.solve()
        """
        while self.solver.iter < self.solver_param.max_iter:
            self.solver.step(1)

            # snapshot
            if self.solver.iter > 0:
                if self.solver.iter % self.solver_param.snapshot == 0:
                    self.solver.snapshot()
                    #fname = self.data_dir + "/" \
                    #        + self.solver_param.snapshot_prefix \
                    #        + ("_%d" % self.solver.iter)
                    #self.solver.net.save("%s.caffemodel" % fname);
                    #iter_fname = self.data_dir + "/" \
                    #        + "iter.npy"
                    #np.save(iter_fname, self.solver.iter)

                # speed show
                if self.solver.iter % self.solver_param.display == 0:
                    t2 = time.time()
                    print 'speed: {:.3f}s / iter'.format((t2-t1)/self.solver_param.display)
                    t1 = t2

                # test
                if self.solver.iter % (self.solver_param.test_interval) == 0:
                    print '#################### test begin ####################'
                    t5 = time.time()
                    loss_len = len(loss)
                    loss_arr = np.zeros(loss_len)
                    for i in range(int(self.solver_param.test_iter[0])):
                        self.prepare_batch_data(i, False)
                        self.solver.test_nets[0].forward()
                        for j in range(loss_len):
                            loss_arr[j] += self.solver.test_nets[0].blobs[loss[j]].data.item()
                    loss_arr /= self.solver_param.test_iter
                    print loss
                    print loss_arr
                    t6 = time.time()
                    print '#################### test finished in %f seconds ####################' % (t6-t5)
                    """

        # after train
        self.solver.net.save(str("%s_finish.caffemodel" % self.solver_param.snapshot_prefix));
        return

    def loadData(self, b_size, callback):
        batch_size, channels, height, width = self.solver.net.blobs['data'].data.shape
        X = np.zeros([b_size, channels, height, width], dtype=np.float32)
        Y = np.zeros([b_size], dtype=np.float32)
        data, y = callback(b_size)
        for i, one in enumerate(data):
            X[i, :, :, :] = one
        for i, one in enumerate(y):
            Y[i] = one
        return X.astype(np.float32), Y.astype(np.float32)

    # should be rewrite
    def loadDataTrain(self, batch_size, offset):
        return

    # should be rewrite
    def loadDataTest(self, batch_size, offset):
        return

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()
        
    def test(self):
        print "debug: ", self.debug
        
def main():
    config = {
        "debug": True,
        "solver": sys.argv[1],
        "pre_model": "",
        "solverstate": "",
        "close_gpu": False,
        "data_dir": "./train_data",
    }
    model = Trainer(config)
    model.test()
    return

if __name__ == "__main__":
    main()
