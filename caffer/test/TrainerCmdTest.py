#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: TrainerCmdTest.py
# Date: 2016 2016年02月29日 星期一 11时27分40秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import sklearn
import sklearn.datasets
import sklearn.linear_model
sys.path.append("/datas/lib/py")
sys.path.append("../lib")
from CaffeTrainerCmd import CaffeTrainerCmd

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class TrainerCmdTest(CaffeTrainerCmd):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: TrainerCmdTest init")
        self.config = config
        self.debug = config["debug"]
        super(TrainerCmdTest, self).__init__(config)
        X, y = sklearn.datasets.make_classification(
                n_samples=200000, n_features=200, 
                n_redundant=0, n_informative=200, 
                n_classes=2, n_clusters_per_class=2, 
                hypercube=False, random_state=0)
        X, Xt, y, yt = sklearn.cross_validation. \
                train_test_split(X, y, test_size=0.5)

        self.X_train = X
        self.y_train = y
        self.i_train = 0

        self.X_test = Xt
        self.y_test = yt
        self.i_test = 0

        clf = sklearn.linear_model.SGDClassifier(
                loss='log', n_iter=1000, 
                penalty='l2', alpha=1e-3, class_weight='auto')
        print "training by linear..."
        #clf.fit(X, y)
        #yt_pred = clf.predict(Xt)
        #print('Accuracy: {:.3f}' \
        #        .format(sklearn.metrics.accuracy_score(yt, yt_pred)))
        return
    
    def dataSrc(self, i, bsize):
        return self.X_train[0:bsize], self.y_train[0:bsize]

    def dataSrcTest(self):
        return self.X_test, self.y_test

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": False,
            "test_iter": 10,
            "cache_dir": "train_test_cache",
            "model_prototxt": "train_val.prototxt",
            "solver_prototxt_template": "solver.template",
            "pre_model": "",
            "batch_size": 100000,
            "train_batch_size": 100,
            "data_use_dir": "/tmp/caffe_train",
            "h5_file": "trainer_cmd",
            "max_iter": 100,
            "data_len": 10000000,
            }
    t = TrainerCmdTest(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

