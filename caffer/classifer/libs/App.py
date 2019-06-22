#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: App.py
# Date: 2016 2016年05月06日 星期五 14时45分42秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time, shutil, datetime
import json, random
import numpy as np
import caffe
import math
import h5py
#from sklearn.cluster import KMeans
import google.protobuf as pb2
sys.path.append("/datas/lib/py")
from lib.Util import *

from DataFactory import DataFactory
from Base import Base

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class App(Base):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: App init")
        self.config = config
        self.debug = config["debug"]
        super(App, self).__init__(config)
        self.is_train = True
        if config.has_key("is_train"):
            self.is_train = config["is_train"]
        self.initApp()
    
    def mkData(self):
        return X, y, Xt, yt

    def mkNet(self, margin=1, dropout=0.5):
        return

    def run(self):
        self.runBase()

    def runBase(self, queue=["maxdata", "convergency", "tearing", "ignore"]):
        # load the datas
        self.loadDatas()

        # do the jobs
        self.log("jobs: " + str(queue))
        for one in queue:
            if self.queue.has_key(one):
                self.queue[one]()

        # close the data loader
        self.data_handler.stopData()
        return

    def initApp(self):
        self.queue = {
                "maxdata": self.maxDataSteps, 
                "convergency": self.convergenceSteps, 
                "dropmax": self.dropmaxSteps,
                "tearing": self.tearingSteps, 
                "ignore": self.ignoreSteps,
                "rst": self.resetSteps,
                }
        self.gpu_id = int(self.config["gpu_id"])
        self.max_iter = self.config["max_iter"]
        if self.config.has_key("continue_max_iter"):
            self.continue_max_iter = self.config["continue_max_iter"]
        else:
            self.continue_max_iter = 2000000
        if self.config.has_key("dropout"):
            self.dropout = self.config["dropout"]
        else:
            self.dropout = 0.7

        if self.is_train:
            self.log_f = open("gpu%d_train.log" % self.gpu_id, "w")
            self.cmdControlInit()

        self.data_factory = DataFactory(self.config)
        self.loadDatas()

        self.state_fname = "gpu%d_states.json" % self.gpu_id
        if os.path.exists(self.state_fname):
            self.states = self.loadOne(self.state_fname)
            if self.config['model'] and self.is_train:
                self.updateStates({"model": self.config['model']})
        else:
            self.states = {
                'lr': self.config['lr'],
                'margin': self.config['margin'],
                'model': self.config['model'],
                'lastmax': self.max_iter,
                'best_acc': 0,
                }
        self.netTester()
        self.best_dir = "./best/"
        if not os.path.exists(self.best_dir):
            os.mkdir(self.best_dir)
        return

    def cmdControlInit(self):
        self.manual_fcmd = "gpu%d_force.cmd" % self.gpu_id
        data = {
                "end": 0, 
                "test": 0,
                "lr": 0,
                "gamma": 0,
                "margin": 0,
                }
        self.saveOne(self.manual_fcmd, data)
        return

    def cmdControl(self, param):
        cmd = self.loadOne(self.manual_fcmd)

        if cmd["margin"] > 0:
            self.updateStates({"margin": cmd["margin"]})

        if cmd["lr"] > 0:
            self.updateStates({"lr": cmd["lr"]})

        if cmd["gamma"] > 0:
            self.config["gamma"] = cmd["gamma"]

        if cmd["test"] > 0:
            s = self.snapshot(*param)
            self.updateStates({"model": s})
            self.testSteps(self.states["model"])

        if cmd["end"] > 0:
            s = self.snapshot(*param)
            self.updateStates({"model": s})
            self.data_handler.stopData()
            print "ended by hand"
            exit()

        self.cmdControlInit()
        return

    def maxDataSteps(self):
        is_max_data = False
        while not is_max_data:
            is_max_data = self.data_handler.addDataSet()
        return

    def netTester(self):
        self.netname = self.mkNet(1, 0.5)
        self.solver_name = self.mkSolver(0.01, self.netname)
        self.initCaffe(self.solver_name, "")
        del self.solver
        return

    def debugLine(self):
        return

    def trainSteps(self, lr, margin, model_snapshot, acc_min=0, msg_prefix="", dropout=0.50):
        self.log(msg_prefix + "training: %.10f, %d, %s, %.2f" % (lr, margin, model_snapshot, acc_min))
        self.log(msg_prefix + "training: data length(%d)" % self.data_handler.useDataLen())
        self.netname = self.mkNet(margin, dropout)
        self.solver_name = self.mkSolver(lr, self.netname)
        self.initCaffe(self.solver_name, str(model_snapshot))
        t1 = time.time()
        finish = False
        acc = 0
        loss = 0
        
        min_loss = 10000
        min_loss_ct = 0
        while self.solver.iter < self.solver_param.max_iter:
            batch = self.data_handler.loadBaches()
            X = np.array(batch[0]).astype(np.float32)
            y = np.array(batch[1]).astype(np.float32)
            nlen = len(y)
            assert nlen % self.train_batch_size == 0

            self.solver.net.set_input_arrays(X, y)
            step_num = nlen / self.train_batch_size

            for i in range(step_num):
                self.solver.step(1)
                self.debugLine()
                acc += self.solver.net.blobs['acc_1'].data / self.config["threshold_check"]
                loss += self.solver.net.blobs['loss_1'].data / self.config["threshold_check"]

            if self.solver.iter % self.config["threshold_check"] == 0:
                t2 = time.time()
                self.log(msg_prefix + "training lr: %.7f" % lr)
                self.log(msg_prefix + "training iter: %d" % self.solver.iter)
                self.log(msg_prefix + 'training speed: {:.3f}s / iter'.format((t2-t1)/self.config["threshold_check"]))
                t1 = t2
                self.log(msg_prefix + "training acc: %.3f" % acc)
                self.log(msg_prefix + "training loss: %.7f" % loss)
                if acc < acc_min: # got the crash error
                    break
                if self.solver.iter >= self.solver_param.max_iter - 1:
                    finish = True
                    self.updateStates({"lastmax": self.solver_param.max_iter + self.max_iter})
                    break
                if loss < min_loss or loss > 7 * min_loss:
                    min_loss = loss
                    min_loss_ct = 0
                else:
                    min_loss_ct += 1
                    if loss > min_loss * 2:
                        min_loss = min_loss + (loss - min_loss) / (min_loss_ct+1)
                    self.log(msg_prefix + "min loss: %.7f, continue(%d)" % (min_loss, min_loss_ct))
                if min_loss_ct > self.config["continue_sameloss_breaker"]:
                    finish = True
                    break
                if acc > self.config["threshold_acc"] and loss < self.config['threshold_loss'] and min_loss_ct >= 5:
                    finish = True
                    break
                # force stop
                self.cmdControl([lr, margin, self.data_handler.useDataLen(), acc, 999, ""]) # means test snapshot

                # init acc and loss
                acc = 0
                loss = 0

        if finish:
            model_snapshot = self.snapshot(lr, margin, self.data_handler.useDataLen(), acc, loss, model_snapshot)

        del self.solver
        return finish, acc, loss, model_snapshot

    def testSteps(self, model, nozero=False):
        assert len(model) > 0

        self.log("testing.........")
        m_name, s_name, crr_iter = self.modelName(model)
        self.test_net = caffe.Net(self.netname, str(m_name), caffe.TEST)
        batch_test = self.data_handler.loadTests()
        Xt = batch_test[0].astype(np.float32)
        yt = batch_test[1].astype(np.float32)
        yt_ = batch_test[2].astype(np.float32)

        self.test_net.set_input_arrays(Xt, yt)

        len_y = len(yt)
        self.log("testing set len: " + str(len_y))
        len_data = int(len_y / self.train_batch_size)
        acc = 0
        loss = 0
        counter = {}
        real_counter = {}
        check = 0
        check_real = 0
        check_real_all = 0
        check_X = 0
        profit = 0
        potential_profit = 0
        profit_amount = 1000
        calc_profit = False
        csv_data = []
        if len(yt_) == len_y:
            calc_profit = True
        for i in range(len_data):
            self.test_net.forward()
            #print self.test_net.blobs["out_1"].data.shape
            for j in range(self.train_batch_size):
                X = self.test_net.blobs["data"].data[j]
                one = self.test_net.blobs["out_1"].data[j]
                label = self.test_net.blobs['label'].data[j]
                l = one.argmax()
                if nozero:
                    l = 1 if one[1] > one[2] else 2
                l = 1 if l == 1 else 0
                #print label, l
                if l == label:
                    check += 1
                if counter.has_key(l):
                    counter[l] += 1
                else:
                    counter[l] = 1

                k = i * self.train_batch_size + j

                # calc the profit
                if calc_profit:
                    l_real = float(yt_[k])
                    if l_real != 0:
                        if l == 1:
                            profit += l_real
                            potential_profit += abs(l_real)
                            profit_amount = profit_amount * (1 + l_real)
                            check_real_all += 1
                            if l_real > 0:
                                check_real += 1
                        elif l == 2:
                            profit -= l_real
                            potential_profit += abs(l_real)
                            profit_amount = profit_amount * (1 - l_real)
                            check_real_all += 1
                            if l_real < 0:
                                check_real += 1
                        csv_data.append([profit, l_real, l, profit_amount])

                #print X.shape, Xt[k].shape
                if np.sum(X - Xt[k]) == 0:
                    check_X += 1
                y = str(int(yt[k]))
                if real_counter.has_key(y):
                    real_counter[y] += 1
                else:
                    real_counter[y] = 1

            acc_one = self.test_net.blobs["acc_1"].data
            loss_one = self.test_net.blobs["loss_1"].data
            acc += acc_one
            loss += loss_one
            #print acc_one, loss_one

        loss /= 1.0 * len_data
        acc /= 1.0 * len_data
        self.log("testing info: " + str(counter))
        self.log("testing result: acc(%.3f) loss(%.7f)" % (acc, loss))
        self.log("testing check: %.3f" % (check * 1.0 / len_y))
        self.log("real check: %.3f" % (check_X * 1.0 / len_y))
        self.log("real info: " + str(real_counter))
        if calc_profit and check_real_all > 0:
            acc = check_real * 1.0 / check_real_all
            self.log("realy check: %.3f" % acc)
            self.log("profit: " + str(profit))
            self.log("profit potential: " + str(potential_profit))
            pacc = profit * 1.0 / potential_profit
            self.log("profit potential per: " + str(pacc))
            self.log("profit amount: " + str(profit_amount))
            fname = "gpu%d_test_res.csv" % self.gpu_id
            mkCsvFileSimple(fname, csv_data)

        if acc > self.getStatus("best_acc") and self.is_train:
            self.updateStates({"best_acc": acc})
            crr_t = datetime.datetime.now().isoformat()
            shutil.copyfile(
                    m_name, 
                    "%s%s_%.04f_%s.caffemodel" % (
                        self.best_dir,
                        m_name.split("/")[-1],
                        acc,
                        crr_t
                        )
                    )
                    #self.best_dir+m_name.split("/")[-1]+"_"+str(round(acc, 3)) + ".%s.caffemodel" % crr_t)

        del self.test_net
        return

    def snapshot(self, lr, margin, dlen, acc, loss, lastone):
        info = self.solver.snapshot()
        tmp_m_name = self.solver_param.snapshot_prefix + "_iter_%d.caffemodel" % self.solver.iter
        tmp_s_name = self.solver_param.snapshot_prefix + "_iter_%d.solverstate" % self.solver.iter
        self.log("snapshoting: " + tmp_s_name)
        if lastone:
            m_lname, s_lname, crr_iter = self.modelName(lastone)
            if os.path.exists(m_lname):
                os.remove(m_lname)
            if os.path.exists(s_lname):
                os.remove(s_lname)
        self.delPrevSnapshots(self.solver.iter)
        return tmp_m_name

    def delPrevSnapshots(self, crr_iter):
        step = int(self.solver_param.snapshot)
        i = 0
        while i < crr_iter:
            tmp_m_name = self.solver_param.snapshot_prefix + "_iter_%d.caffemodel" % i
            tmp_s_name = self.solver_param.snapshot_prefix + "_iter_%d.solverstate" % i
            if os.path.exists(tmp_m_name):
                os.remove(tmp_m_name)
            if os.path.exists(tmp_s_name):
                os.remove(tmp_s_name)
            i += step
        return

    def modelName(self, s):
        s_arr = s.split(".")
        del s_arr[-1]
        prefix = ".".join(s_arr)
        m_name = prefix + ".caffemodel"
        s_name = prefix + ".solverstate"
        model_iter = int(s_arr[0].split("_")[-1])
        #print m_name
        #print model_iter
        return m_name, s_name, model_iter

    def loadDatas(self):
        self.data_handler = self.data_factory.init(self.config['data_type'])
        lx, lt = self.data_handler.placeAllData(*self.mkData())
        self.log("train len: %d, test len: %d" % (lx, lt))
        return

    def mkSolver(self, lr=0.001, netname="net.prototxt"):
        tpl = self.config["solver_template"]
        fname_arr = tpl.split(".")
        fname_arr.pop()
        fname = ".".join(fname_arr)
        fname_arr = fname.split('/')
        fname = ("gpu%d_" % self.gpu_id) + fname_arr[-1]
        content = getFileContent(tpl, False)
        content = content.replace("<lr>", str(lr))
        content = content.replace("<net>", str(netname))
        content = content.replace("<max_iter>", str(self.states["lastmax"]))
        content = content.replace("<gpu>", str(self.gpu_id))
        writeToFile(fname, content)
        return fname

    def convergenceSteps(self):
        is_convergence = False
        acc_min = 0
        is_max_data = self.data_handler.isMax()
        while not is_max_data or not is_convergence:
            is_end, acc, loss, smodel = self.trainSteps(
                    self.states['lr'], 
                    self.states['margin'], 
                    self.states['model'], 
                    acc_min,
                    "convergency ",
                    self.dropout
                    )
            self.log("convergency training info: acc(%.3f) loss(%.5f)" % (acc, loss))
            if acc > 0.9: # set the min acc
                acc_min = 0.40

            if is_end: # finished train
                if acc < self.config['threshold_acc']: # not learned enough
                    self.updateStates({'lr': self.states['lr'] * self.config['gamma']})
                else: # ok add dataset
                    is_max_data = self.data_handler.addDataSet()
                self.updateStates({"model": smodel})
            else:
                self.updateStates({'lr': self.states['lr'] * self.config["gamma"]})

            if loss < self.config['threshold_loss']:
                is_convergence = True

            # if still not convergenced, just make it convergenced
            if self.config['stop_lr'] > self.states['lr']:
                self.log("convergency: can't be convergenced!!!")
                break
            self.testSteps(smodel)
        self.log("convergency finished.")
        return

    def dropmaxSteps(self): #TODO not tested
        is_convergence = False
        acc_min = 0
        drop_e = self.dropout
        not_enough_ct = 0
        while not_enough_ct < 3:
            is_end, acc, loss, smodel = self.trainSteps(
                    self.states['lr'], 
                    self.states['margin'], 
                    self.states['model'], 
                    acc_min,
                    "dropmax ",
                    dropout=drop_e
                    )
            self.log("dropmax training info: acc(%.3f) loss(%.5f)" % (acc, loss))
            if acc > 0.9: # set the min acc
                acc_min = 0.7

            if is_end: # finished train
                if loss > self.config['threshold_loss']: # not learned enough
                    self.updateStates({'lr': self.states['lr'] * self.config["gamma"]})
                    not_enough_ct += 1
                else: # ok add dropout
                    drop_e += 0.05
                    not_enough_ct = 0
                self.updateStates({"model": smodel})
            else:
                self.updateStates({'lr': self.states['lr'] * self.config["gamma"]})
                not_enough_ct += 1

            if drop_e >= 1:
                break
            self.testSteps(smodel)
        self.log("dropmax finished.")
        return

    def tearingSteps(self):
        is_convergence = False
        acc_min = 0.7
        while not is_convergence:
            is_end, acc, loss, smodel = self.trainSteps(
                    self.states['lr'], 
                    self.states['margin'], 
                    self.states['model'], 
                    acc_min,
                    "tearing "
                    )
            self.log("tearing training info: acc(%.3f) loss(%.5f)" % (acc, loss))
            if is_end:
                if loss > self.config['threshold_loss']: # not learned enough
                    self.updateStates({'lr': self.states['lr'] * self.config["gamma"]})
                else:
                    if self.states["margin"] + 1 <= self.config["margin_max"]:
                        self.updateStates({'margin': self.states['margin'] + 1})
                self.updateStates({"model": smodel})
                if self.states["margin"] + 1 > self.config["margin_max"]:
                    break
            else:
                self.updateStates({'lr': self.states['lr'] * self.config['gamma']})

            # if still not convergenced, just make it convergenced
            if self.config['stop_lr'] > self.states['lr']:
                self.log("tearing: can't be convergenced!!!")
                break
            self.testSteps(smodel)
        self.testSteps(smodel)
        self.log("tearing finished.")
        return

    def ignoreSteps(self):
        self.log("doing the ignoreSteps...")
        self.test_net = caffe.Net("net.prototxt", str(self.states['model']), caffe.TEST)
        datalen = self.data_handler.useDataLen()

        i = 0
        step = self.test_net.blobs["data"].data.shape[0]

        feats_shape = list(self.test_net.blobs["sigmoid_1"].data.shape)
        feats_shape[0] = datalen
        X_feats = np.zeros(tuple(feats_shape))
        y_shape = list(self.test_net.blobs["label"].data.shape)
        y_shape[0] = datalen
        y_all = np.zeros(tuple(y_shape))
        while i < datalen:
            batch = self.data_handler.loadOrderBatchs(i, step)
            X = batch[0].astype(np.float32) 
            y = batch[1].astype(np.float32)
            self.test_net.set_input_arrays(X, y)
            self.test_net.forward()
            feats = self.test_net.blobs["sigmoid_1"].data
            X_feats[i:i+step] = feats
            y_all[i:i+step] = y.reshape(step)
            #print feats
            #print y
            i += step

        self.log("clustering the features...")
        ignore_arr = self.ignoreSpecial(X_feats, y_all, 0.1)
        for i in range(datalen):
            if ignore_arr[i]:
                self.data_handler.ignore(i, self.config['cluster_number'])

        del self.test_net
        return

    def resetSteps(self):
        self.updateStates({"lr": self.config["lr"]})
        return

    def ignoreSpecial(self, X_feats, y, per):
        datalen = len(y)
        clusters = {}
        for i in range(datalen):
            label = str(y[i])
            if clusters.has_key(label):
                clusters[label]["datas"].append(X_feats[i])
            else:
                clusters[label] = {"datas": [X_feats[i]]}

        self.log("data len: "+str(datalen))
        for one_y in clusters.keys():
            self.log("label: " + one_y)
            info = np.array(clusters[one_y]["datas"])
            clusters[one_y]["mean"] = info.mean(axis=0)
            self.log("label len: %d" % len(info))
            #self.log("label mean: " + str(clusters[one_y]["mean"]))

        d = 0
        dmax = 0
        dmin = 1000
        for i in range(datalen):
            label = str(y[i])
            diff = X_feats[i] - clusters[label]["mean"]
            d_one = math.sqrt(diff.dot(diff))
            d += d_one
            if d_one > dmax:
                dmax = d_one
            if d_one < dmin:
                dmin = d_one
        dmean = d/datalen
        self.log("cluster summery: dmean(%.5f), dmax(%.5f), dmin(%.5f)" % (dmean, dmax, dmin))

        ignore_ck = dmax * 0.7
        if dmean * 2 < dmax:
            ignore_ck = dmean * 2

        ignore_indexes = []
        for i in range(datalen):
            label = str(y[i])
            diff = X_feats[i] - clusters[label]["mean"]
            d_one = math.sqrt(diff.dot(diff))
            if d_one > ignore_ck:
                ignore_indexes.append(1)
            else:
                ignore_indexes.append(0)

        self.log("ignore len: %d" % np.array(ignore_indexes).sum())
        return ignore_indexes

    def redefLabel(self, y, y_, threshold=0.0):
        y *= 0.0
        y_.shape = y.shape
        xargs = np.argwhere(y_ > threshold)
        y[xargs] = 1.0
        xargs = np.argwhere(y_ < -threshold)
        y[xargs] = 2.0
        #print y.shape
        labels = {}
        for one in y:
            one = int(one)
            k = str(one)
            if labels.has_key(k):
                labels[k] += 1
            else:
                labels[k] = 1
        self.log("train labels: " + str(labels))
        return y

    def getStatus(self, k):
        if not self.states.has_key(k):
            self.states[k] = 0

        return self.states[k]

    def updateStates(self, kvs):
        for k in kvs.keys():
            self.states[k] = kvs[k]

        self.log("states changed")
        self.log(self.states)

        self.saveOne(self.state_fname, self.states)
        return

    def initCaffe(self, solver, model_snapshot=''):
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        self.solver = caffe.SGDSolver(solver)
        self.solver_param = caffe.proto.caffe_pb2.SolverParameter()
        with open(solver, 'rt') as fd:
            pb2.text_format.Merge(fd.read(), self.solver_param)
        if model_snapshot:
            fsuffix = model_snapshot.split(".")
            m_name, s_name, crr_iter = self.modelName(model_snapshot)
            if os.path.exists(s_name) and crr_iter < self.continue_max_iter:
                self.solver.restore(s_name)
            else:
                self.solver.net.copy_from(m_name)
        self.train_batch_size = self.solver.net.blobs['data'].data.shape[0]
        return

    def testPrint(self):
        print "Hello World!"

    def log(self, msg):
        print msg
        if self.is_train:
            self.log_f.write(mktime("%H:%M:%S")+" "+str(msg)+'\n')
            self.log_f.flush()
        return

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            # training param
            "gpu_id": 0, 
            "lr": 0.001,
            "stop_lr": 0.00000001,
            "margin": 1,
            "model": "", 
            "solver_template": "solver_adadelta.prototxt.template",
            "continue_sameloss_breaker": 20,
            # training thresholds
            "threshold_check": 20000, # the steps to do checking
            "threshold_acc": 0.98, # train set accuracy
            "threshold_loss": 0.1, # train set loss
            # convergence config
            "data_init_len": 300,
            "data_growth": 0.1,
            "data_shape": (1, 2, 4), #TODO
            "batch_len": 32 * 10,
            "data_type": "mem",
            }
    t = App(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

