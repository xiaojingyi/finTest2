#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: TrainerTest.py
# Date: 2016 2016年02月27日 星期六 16时29分39秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from model.Trainer import Trainer
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class TrainerTest(Trainer):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: TrainerTest init")
        self.config = config
        self.debug = config["debug"]
        super(TrainerTest, self).__init__(config)
        X, y = sklearn.datasets.make_classification(
                n_samples=10000, n_features=200, 
                n_redundant=0, n_informative=200, 
                n_classes=2, n_clusters_per_class=2, 
                hypercube=False, random_state=0)
        X, Xt, y, yt = sklearn.cross_validation. \
                train_test_split(X, y, test_size=0.1)

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
    
    def testPrint(self):
        print "Hello World!"

    def loadDataTrain(self, batch_size):
        x = []
        y = []
        for i in range(batch_size):
            x.append(self.X_train[i].reshape((1, 1, 200)))
            y.append(self.y_train[i])
        
        return np.array(x), np.array(y)

    def loadDataTest(self, batch_size):
        x = []
        y = []
        for i in range(batch_size):
            x.append(self.X_test[i].reshape((1, 1, 200)))
            y.append(self.y_test[i].reshape((1, 1, 1)))

        return np.array(x), np.array(y)

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    config = {
        "debug": True,
        "solver": "solver.prototxt",
        "pre_model": "",
        "solverstate": "",
        "close_gpu": True,
        "data_dir": "./tmp",
    }
    t = TrainerTest(config)
    t.prepare_data(9000, 1000)
    t.train(["loss"])
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

