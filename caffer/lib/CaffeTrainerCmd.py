#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: CaffeTrainerCmd.py
# Date: 2016 2016年02月28日 星期日 12时14分42秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import threading
import random
import h5py
import json
import math
from threading import Thread
sys.path.append("/datas/lib/py")
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class CaffeTrainerCmd(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: CaffeTrainerCmd init")
        self.config = config
        self.debug = config["debug"]

        # config
        self.test_iter = config['test_iter']
        self.cache_dir = config['cache_dir']
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.model_prototxt = config['model_prototxt']
        self.solver_prototxt_template = config['solver_prototxt_template']
        self.pre_model = config['pre_model']
        self.batch_size = config['batch_size']
        self.train_batch_size = config['train_batch_size']
        self.data_gen_dir = config['data_gen_dir']
        self.data_use_dir = config['data_use_dir']
        self.h5_file = config['h5_file']
        self.h5_test_file = config['h5_file'] + "_test"
        self.max_iter = config['max_iter']
        self.data_len = config['data_len']
        self.review_time = config['review_time']

        # init val
        self.is_gening_data = False
        self.is_training = False
        self.is_copying = False
        self.is_data = False
        self.is_copyed = False
        self.stop = False

    ######## start this should be rewrite ########

    def dataSrc(self, i, bsize):
        return 

    def dataSrcTest(self):
        return 

    ######## end this should be rewrite ########

    def run(self):
        t = threading.Thread(target = self.prepareData)
        t.start()
        self.train()
        t.join()
        return

    def prepareCp(self):
        if self.is_copying == False:
            if self.is_training == False and self.is_data == True: 
                t = time.time()
                print "copying data..."
                self.is_copying = True
                orig_file = self.data_gen_dir + "/" + self.h5_file
                run_file = self.data_use_dir + "/" + self.h5_file
                orig_file_test = self.data_gen_dir + "/" + self.h5_test_file
                run_file_test = self.data_use_dir + "/" + self.h5_test_file

                cmd = "rm /tmp/caffe.* -rf"
                self.call(cmd)

                cmd = "rm -rf %s" % run_file
                self.call(cmd)
                cmd = "rm -rf %s" % run_file_test
                self.call(cmd)

                cmd = "cp %s %s" % (orig_file, run_file)
                self.call(cmd)
                cmd = "cp %s %s" % (orig_file_test, run_file_test)
                self.call(cmd)

                cmd = "rm -rf " + orig_file
                self.call(cmd)
                cmd = "rm -rf " + orig_file_test
                self.call(cmd)

                self.is_copying = False
                self.is_copyed = True
                self.is_data = False
                print "used: %f s" % (time.time() - t)
        return

    def prepareData(self):
        i = 0
        ifile = self.cache_dir+"/data_i.npy"
        if os.path.exists(ifile):
            i = np.load(ifile)

        train_fls = './train.txt'
        test_fls = './test.txt'
        if not os.path.exists(train_fls):
            writeToFile(train_fls, self.data_use_dir + "/" + self.h5_file)
        if not os.path.exists(test_fls):
            writeToFile(test_fls, self.data_use_dir + "/" + self.h5_test_file)

        while True:
            if self.stop:
                break
            if self.is_gening_data == False:
                if self.is_copying == False and self.is_data == False:
                    print "gening data(%d)..." % i
                    t = time.time()
                    self.is_gening_data = True
                    x, y = self.dataSrc(i, self.batch_size)
                    self.mkH5Set(x, y, self.h5_file)
                    del x
                    del y
                    x_, y_ = self.dataSrcTest()
                    self.mkH5Set(x_, y_, self.h5_test_file)
                    del x_
                    del y_
                    self.is_gening_data = False
                    self.is_data = True
                    i += 1
                    if i >= int(math.ceil((self.data_len - 1) / self.batch_size)):
                        i = 0
                    np.save(ifile, i)
                    print "used: %f s" % (time.time() - t)
            time.sleep(0.1)
        return

    def train(self):
        i = 0
        ifile = self.cache_dir+"/train_i.npy"
        if os.path.exists(ifile):
            i = np.load(ifile)

        while True:
            if self.stop:
                break
            self.prepareCp()
            if self.is_training == False:
                if self.is_copyed == True and self.is_copying == False:
                    t = time.time()
                    self.is_training = True
                    self.trainIter(i)

                    self.is_training = False
                    self.is_copyed = False
                    np.save(ifile, i)
                    i += 1
                    print "used: %f s" % (time.time() - t)
            if i >= self.max_iter:
                self.stop = True
            time.sleep(0.1)
        return

    def mkSolverPrototxt(self, i):
        assert(os.path.exists(self.solver_prototxt_template))

        fpath = os.path.dirname(self.solver_prototxt_template)
        if not fpath:
            fpath = "./"
        self.solver_prototxt = fpath + "/solver.prototxt"
        fp = open(self.solver_prototxt_template, "r")
        content = fp.read()
        fp.close()
        content = content.replace("{max_iter}", str((i+1) * int(self.review_time * self.batch_size / self.train_batch_size + 1)))
        #content = content.replace("{batch_size}", str(self.train_batch_size))
        writeToFile(self.solver_prototxt, content)
        return self.solver_prototxt

    def trainIter(self, time_iter):
        # make solver.prototxt
        self.mkSolverPrototxt(time_iter)

        # check if should test
        if time_iter > 0 and time_iter % self.test_iter == 0:
            print "testing..."
            model = self.getSnapshot(self.cache_dir).replace("solverstate", "caffemodel")
            cmd = "caffe test -model %s -weights %s \
                    -iterations 100 -gpu all  2>&1 \
                    | grep -E \"solver.cpp|caffe.cpp\" \
                    >> %s/test.log" % (
                            self.model_prototxt, 
                            model,
                            self.cache_dir
                            )
            self.call(cmd)

        # training
        cmd = "caffe train -solver %s " % self.solver_prototxt
        if time_iter == 0 and self.pre_model:
            cmd += " -weights %s " % self.pre_model
        elif time_iter > 0:
            cmd += " -snapshot %s " % self.getSnapshot(self.cache_dir)
        cmd += " -gpu all  2>&1 | grep -E \"solver.cpp|caffe.cpp\" >> \
                %s/train.log " % (self.cache_dir)
        self.call(cmd)
        #cmd = "echo 1 > /proc/sys/vm/drop_caches"
        #self.call(cmd)
        print "train(%d) finished." % time_iter
        return

    def getSnapshot(self, run_dir):
        ls = os.listdir(run_dir)
        len_max = 0
        str_max = ""
        for one in ls:
            if one.split(".")[-1] == "solverstate":
#                print one
                if len(one) >= len_max:
                    len_max = len(one)
                    if one >= str_max or len(str_max) < len_max:
                        str_max = one

        for one in ls:
            if one.split(".")[-1] == "solverstate":
                cmd = "rm -rf %s/%s  %s/%s" % (run_dir, one, run_dir, one.replace("solverstate", "caffemodel"))
                if len(str_max) > 0 and one != str_max:
                    self.call(cmd)
                
        return "%s/%s" % (run_dir, str_max)
    
    def mkH5Set(self, x, y, fname):
        fpath = self.data_gen_dir + "/" + fname
        comp_kwargs = {}
        with h5py.File(fpath, 'w') as f:
            f.create_dataset('data', data=x.astype(np.float32), **comp_kwargs)
            f.create_dataset('label', data=y.astype(np.float32), **comp_kwargs)
        return
            
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

    def call(self, cmd):
        print cmd
        if not self.debug:
            exeCmd(cmd)

def main():
    conf = {
            "debug": True,
            }
    t = CaffeTrainerCmd(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

