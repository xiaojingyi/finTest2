import os.path
#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-8-14 17:18:53$
# Note: This source file is NOT a freeware
# Version: main.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-8-14 17:18:53$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
from lib.Util import *
import threading
from threading import Thread
import numpy as np
from DeepData import DeepData
from Temp import Temp

class Trainer(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: main init")
        self.config = config
        self.debug = config["debug"]
        self.cmd = {
            "gen_data": True,
        }
        self.status = {
            "all": "stop", # running
            "iter_all": 0,
            "iter_data": 0,
        }
        self.data = DeepData(self.config['data'])
        #super(main, self).__init__(config)
        
    def dataGen(self):
        while self.cmd['gen_data']:
            res = self.data.run()
            if self.debug:
                print res
            time.sleep(1)
        return
    
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

    def call(self, cmd):
        print cmd
        if not self.debug:
            exeCmd(cmd)
    
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
    
    def train(self, i):
        while self.data.cp_st == 0:
            print "wait cp finished..."
            time.sleep(1)
        
        if i > 0 and i % self.config['runner']['test_iter'] == 0:
            print "testing..."
            model = self.getSnapshot(self.config['runner']['cache']).replace("solverstate", "caffemodel")
            cmd = "caffe test -model %s -weights %s -iterations 100 -gpu all  2>&1 | grep -E \"solver.cpp|caffe.cpp\" >> test.log" % (self.config['runner']["model"], model)
            self.call(cmd)
            
        print "trainning..."
        cmd = "caffe train -solver %s " % self.config['runner']['solver']
        if i == 0 and self.config['runner']['brain']:
            cmd += " -weights %s " % self.config['runner']['brain']
        elif i > 0:
            cmd += " -snapshot %s " % self.getSnapshot(self.config['runner']['cache'])
        cmd += " -gpu all  2>&1 | grep -E \"solver.cpp|caffe.cpp\" >> train.log "
        self.call(cmd)

        cmd = "rm -rf %s %s " % (self.config['runner']['db'], self.config['runner']['db']+"_test")
        self.call(cmd)
        self.data.cp_st = 0
        print "train(%d) finished." % i
        return
        
    def run(self):
        t = threading.Thread(target = self.dataGen)
        t.start()
        iter = 0
        template = Temp({"debug": True,})
        try:
            index = np.load("cache/global_iter.npy")
        except:
            index = -1
        for i in range(self.config["max_iter"]):
            if index >= i:
                continue
            template.mkSolver(i, self.config['runner']['iter_step'])
            self.train(i)
            np.save("cache/global_iter.npy", i)
            time.sleep(1)
        self.cmd['gen_data'] = False
        t.join()
        
def main():
    db = "/tran_tmp/caffedb"
    config = {
        "debug": False,
        "data": {
#            "data_src": "lfwset_train/lfwset/",
#            "test_src": "lfwset_val/attset/",
            "data_src": "attset_train/attset/", 
            "test_src": "attset_val/attset/",
            "train_dir_pre": "./cache/caffedb",
            "test_db": "./cache/caffedb_test",
            "train_dir": db,
            "db_size": 400,
            "iter_size": 100,
            "width": 227,
            "height": 227,
        },
        "runner": {
            "db": db,
            "test_iter": 10,
            "iter_step": 120,
            "cache": "cache",
            "solver": "run/solver.prototxt",
            "model": "run/sauron_tree.prototxt",
            "brain": False,
        },
        "max_iter": 100000,
    }
    model = Trainer(config)
    model.run()
    return

if __name__ == "__main__":
    main()
