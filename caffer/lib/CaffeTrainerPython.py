#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: CaffeTrainerPython.py
# Date: 2016 2016年04月21日 星期四 14时49分40秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import caffe
import google.protobuf as pb2
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class CaffeTrainerPython(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: CaffeTrainerPython init")
        self.config = config
        self.debug = config["debug"]
        #super(CaffeTrainerPython, self).__init__(config)
        self.init()
    
    def dataLoader(self, nlen):
        return (X, y), (Xt, yt), nlen

    def init(self):
        caffe.set_device(self.config['gpu_id'])
        caffe.set_mode_gpu()
        self.solver = caffe.SGDSolver(self.config["solver"])
        self.solver_param = caffe.proto.caffe_pb2.SolverParameter()
        with open(self.config["solver"], 'rt') as fd:
            pb2.text_format.Merge(fd.read(), self.solver_param)
        if self.config.has_key("weights") and self.config["weights"] is not None:
            fsuffix = self.config["weights"].split(".")
            if fsuffix[-1] == "solverstate":
                self.solver.restore(self.config["weights"])
            elif fsuffix[-1] == "caffemodel":
                self.solver.net.copy_from(self.config["weights"])
        self.train_batch_size = self.solver.net.blobs['data'].data.shape[0]
        self.test_batch_size = self.solver.test_nets[0].blobs['data'].data.shape[0]
        return

    def run(self):
        t1 = time.time()
        loss = 0
        acc = 0
        while self.solver.iter < self.solver_param.max_iter:
            Xy, Xyt, nlen = self.dataLoader(self.config["nbatch"])
            assert nlen % self.train_batch_size == 0

            self.solver.net.set_input_arrays(*Xy)
            self.solver.test_nets[0].set_input_arrays(*Xyt)

            step_num = nlen / self.train_batch_size
            for i in range(step_num):
                self.solver.step(1)
                one_loss = self.solver.net.blobs['loss_1'].data
                one_acc = self.solver.net.blobs['acc_1'].data
                #print one_loss, one_acc
                loss += one_loss / self.solver_param.display
                acc += one_acc / self.solver_param.display

            if self.solver.iter % self.solver_param.display == 0:
                t2 = time.time()
                print 'loss:', loss
                print 'acc:', acc
                loss = 0
                acc = 0
                print 'speed: {:.3f}s / iter'.format((t2-t1)/self.solver_param.display)
                t1 = t2
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
    t = CaffeTrainerPython(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

