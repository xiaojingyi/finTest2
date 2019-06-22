#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Tester.py
# Date: 2016 Wed 05 Oct 2016 11:50:39 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import numpy as np
sys.path.append("/datas/lib/py")
sys.path.append("../libs")
from Runner import Runner
from sklearn.cluster import KMeans
#from sklearn.utils.validation import check_array

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Tester(Runner):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Tester init")
        self.config = config
        self.debug = config["debug"]
        super(Tester, self).__init__(config)
        self.model = None
        self.mres = []
        self.cluster_N = 100
    
    def afterTest(self, features, labels):
        return

    # features sorted test
    def afterTest(self, features, labels):
        print features.shape, labels.shape
        arr = np.concatenate((features, labels), axis=1)
        print arr.shape
        len_arr = arr.shape[0]
        len_use = int(len_arr * 0.1)
        for i in range(features.shape[1]):
            tmp = np.array(sorted(
                arr, 
                key=lambda x: x[i], 
                reverse=True))
            print tmp[0:len_use, 10].sum() / abs(tmp[0:len_use, 10]).sum()

    # KMeans test
    def afterTest1(self, features, labels):
        if self.model == None:
            self.model = KMeans( n_clusters=self.cluster_N, n_jobs=4).fit(features)
        labels_predict = self.model.predict(features)
        l = [0] * self.cluster_N
        l_ct = [0] * self.cluster_N
        l_walk = [0] * self.cluster_N
        l_walk_sum = [0] * self.cluster_N

        for i in range(len(features)):
            label = labels_predict[i]
            walk = labels[i][2]
            if walk > 0:
                l[label] += 1
                l_walk[label] += walk
            l_ct[label] += 1
            l_walk_sum[label] += abs(walk)

        ct_up = ct_all = 0
        walk_up = walk_all = 0
        if len(self.mres) == 0:
            scores = [ l[i] * 1.0 / l_ct[i] 
                    for i in range(self.cluster_N) ]
            """
            scores = [ l_walk[i] * 1.0 / l_walk_sum[i] 
                    for i in range(self.cluster_N) ]
            scores.sort(reverse=True)
                if walk_acc < threshold:
            """
            scores.sort()
            threshold = scores[(int)(self.cluster_N * 0.1)]
            for i in range(self.cluster_N):
                ct_acc = l[i] * 1.0 / l_ct[i]
                walk_acc = l_walk[i] * 1.0 / l_walk_sum[i]
                if walk_acc < threshold:
                    self.mres.append(1)
                    #print l[i], l_ct[i], walk_acc, ct_acc
                    ct_up += l[i]
                    ct_all += l_ct[i]
                    walk_up += l_walk[i]
                    walk_all += l_walk_sum[i]
                else:
                    self.mres.append(0)
        else:
            for i in range(self.cluster_N):
                ct_acc = l[i] * 1.0 / l_ct[i]
                walk_acc = l_walk[i] * 1.0 / l_walk_sum[i]
                if self.mres[i] == 1:
                    #print l[i], l_ct[i], walk_acc, ct_acc
                    ct_up += l[i]
                    ct_all += l_ct[i]
                    walk_up += l_walk[i]
                    walk_all += l_walk_sum[i]
        print ct_all, "ct_acc: ", ct_up * 1.0 / ct_all
        print walk_all, "walk_acc: ", walk_up * 1.0 / walk_all
        return

    def run(self):
        self.testSteps(self.config["model"], "sigmoid_4")
        self.testSteps(self.config["model"], "sigmoid_4")
        self.testSteps(self.config["model"], "sigmoid_4")
        return

def main():
    conf = {
            "debug": True,
            "is_train": False,
            # training param
            "gpu_id": sys.argv[2],
            "max_iter": 10000000,
            "lr": 0.00001,
            "gamma": 1.0,
            "dropout": 0.5,
            "stop_lr": 0.000001,
            "margin": 1,
            "margin_max": 127,
            "model": sys.argv[1],
            #"solver_template": "solver_rmsprop.prototxt.template",
            "solver_template": "solver_adagrad.prototxt.template",
            #"solver_template": "solver_adam.prototxt.template",
            "continue_sameloss_breaker": 17,

            # training thresholds
            "threshold_check": 100, # the steps to do checking
            "threshold_acc": 0.98, # train set accuracy
            "threshold_loss": 0.0001, # train set loss
            }
    t = Tester(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

