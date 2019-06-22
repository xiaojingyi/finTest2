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
import google.protobuf as pb2
sys.path.append("/datas/lib/py")
from lib.Util import *

from TestCounter import TestCounter
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
        if config.has_key("out_blob"):
            self.out_blob = config["out_blob"]
        else:
            self.out_blob = "out_1"
        if config.has_key("test_len"):
            self.test_len = config["test_len"]
        else:
            self.test_len = 10240 * 8
        self.initApp()
        self.test_iter = 0
        self.out_index = 1
        if config.has_key("loss_index"):
            self.loss_index = config["loss_index"]
        else:
            self.loss_index = 1
    
    def mkNet(self, margin=1, dropout=0.5):
        return

    def run(self):
        self.runBase()

    def runBase(self, queue=["convergency", ]):
        # do the jobs
        self.log("jobs: " + str(queue))
        for one in queue:
            if self.queue.has_key(one):
                self.queue[one]()

        return

    def initApp(self):
        self.queue = {
                "convergency": self.convergenceSteps, 
                }

        self.gpu_id = int(self.config["gpu_id"])
        self.max_iter = self.config["max_iter"]
        self.test_net = None
        self.solver = None
        if self.config.has_key("dropout"):
            self.dropout = self.config["dropout"]
        else:
            self.dropout = 0.7

        if self.is_train:
            self.log_f = open("gpu%d_train.log" % self.gpu_id, "w")
            self.cmdControlInit()

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

    def cmdControlInit(self): #TODO
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
            self.log("ended by hand")
            exit()

        self.cmdControlInit()
        return

    def netTester(self):
        self.netname, self.netname_test = self.mkNet(1, 0.5)
        self.solver_name = self.mkSolver(0.01, self.netname)
        self.initCaffe(self.solver_name, "")
        del self.solver
        self.solver = None
        return

    def debugLine(self):
        return

    def trainSteps(self, lr, margin, model_snapshot, acc_min=0, msg_prefix="", dropout=0.50):
        # first output the starting message
        self.log(msg_prefix + "training: %.10f, %d, %s, %.2f" % (lr, margin, model_snapshot, acc_min))

        # make and init the net
        self.netname, self.netname_test = self.mkNet(margin, dropout)
        self.solver_name = self.mkSolver(lr, self.netname)
        self.initCaffe(self.solver_name, str(model_snapshot))

        # starting to train
        t1 = time.time()
        finish = False
        acc = 0
        loss = 0
        
        min_loss = 10000
        min_loss_ct = 0
        while self.solver.iter < self.solver_param.max_iter:
            self.solver.step(1)
            self.debugLine()
            if self.solver.net.blobs.has_key('acc_'+str(self.out_index)):
                acc += self.solver.net.blobs['acc_'+str(self.out_index)].data / self.config["threshold_check"]
            loss += self.solver.net.blobs['loss_'+str(self.loss_index)].data / self.config["threshold_check"]

            if self.solver.iter % self.config["threshold_check"] == 0:
                t2 = time.time()
                # training info
                self.log(msg_prefix + "training out index: %d" % self.out_index)
                self.log(msg_prefix + "training lr: %.7f" % lr)
                self.log(msg_prefix + "training iter: %d" % self.solver.iter)
                self.log(msg_prefix + 'training speed: {:.3f}s / iter'.format((t2-t1)/self.config["threshold_check"]))
                self.log(msg_prefix + "training acc: %.3f" % acc)
                self.log(msg_prefix + "training loss: %.7f" % loss)

                t1 = t2
                if acc < acc_min: # got the crash error
                    break

                # the last iter
                if self.solver.iter >= self.solver_param.max_iter - 1:
                    finish = True
                    self.updateStates({"lastmax": self.solver_param.max_iter + self.max_iter})
                    break

                # min lose adapt
                if loss < min_loss or loss > 7 * min_loss:
                    min_loss = loss
                    min_loss_ct = 0
                else:
                    min_loss_ct += 1
                    if loss > min_loss * 2:
                        min_loss = min_loss + (loss - min_loss) / (min_loss_ct+1)
                    self.log(msg_prefix + "min loss: %.7f, continue(%d)" % (min_loss, min_loss_ct))

                # the continue min loss count reached
                if min_loss_ct > self.config["continue_sameloss_breaker"]:
                    finish = True
                    break

                # reach the convergenced
                if acc > self.config["threshold_acc"] and loss < self.config['threshold_loss'] and min_loss_ct >= 5:
                    finish = True
                    break

                # force stop
                self.cmdControl([lr, margin, acc, 999, ""]) # means test snapshot

                # init acc and loss
                acc = 0
                loss = 0

        if finish:
            model_snapshot = self.snapshot(lr, margin, acc, loss, model_snapshot)

        #del self.solver
        return finish, acc, loss, model_snapshot

    def testCallback(self, blobs, bsize, i=0, i_max=0):
        res = []
        for j in range(bsize):
            X = blobs["data"].data[j]
            one = blobs["out_1"].data[j]
            label = one.argmax()
            label_real = blobs["label"].data[j]
            walk = blobs["label_"].data[j]
            res.append([label, int(sum(label_real)), sum(walk)])
        return res

    def afterTest(self, features, labels):
        self.log("features " + str(features.shape))
        self.log("labels " + str(labels.shape))
        return

    def testSteps(self, model, featue_index=""):
        assert len(model) > 0
        if featue_index:
            self.out_blob = featue_index

        self.log("testing.........")
        m_name, s_name, crr_iter = self.modelName(model)
        if self.test_net == None:
            self.test_net = caffe.Net(self.netname_test, str(m_name), caffe.TEST)
        else:
            self.test_net.copy_from(str(m_name))
        dlen = self.test_len
        bsize = self.test_net.blobs['data'].data.shape[0]
        loop_max = int(dlen / bsize)
        counter = TestCounter({
            "debug": False, 
            "fname": "gpu%d_test_res.csv" % self.gpu_id,
            "stop_lose": 100000,
            })
        real_l = [0, 0, 0,]
        features = None
        labels = None
        for i in range(loop_max):
            self.test_net.forward()
            b_res = self.testCallback(self.test_net.blobs, bsize, i, loop_max)
            if i == 0:
                features = self.test_net.blobs[self.out_blob].data
                labels = np.array(b_res)
            else:
                features = np.concatenate((
                    features,
                    self.test_net.blobs[self.out_blob].data,
                    ))
                labels = np.concatenate((
                    labels,
                    np.array(b_res),
                    ))
            for one in b_res:
                l, l_real, l_ = one
                counter.addRes(l, l_)
                real_l[l_real] += 1
            if i % (loop_max/10) == 0:
                print "finished:", i, loop_max
        counter.saveRes()
        if features != None:
            self.afterTest(features, labels)
        label_info, acc_info = counter.getStatus()
        self.log("testing labels: " + str(label_info))
        self.log("testing real labels: " + str(real_l))
        self.log("testing res: " + str(acc_info))
        if len(acc_info) > 0:
            acc = acc_info[-1] # use acc/std
            if label_info['1'] + label_info['2'] > 1024 \
                    and acc > self.getStatus("best_acc") \
                    and self.is_train:
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
            elif acc <= self.getStatus("best_acc"):
                self.updateStates({
                    "best_acc": self.getStatus("best_acc") * 0.999
                    })

        self.test_iter += 1
        return

    def snapshot(self, lr, margin, acc, loss, lastone):
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
        step = 10
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
        return m_name, s_name, model_iter

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
        content = content.replace("<max_iter>", str(self.max_iter))
        content = content.replace("<gpu>", str(self.gpu_id))
        writeToFile(fname, content)
        return fname

    def convergenceSteps(self):
        is_convergence = False
        acc_min = 0
        while not is_convergence:
            is_end, acc, loss, smodel = self.trainSteps(
                    self.states['lr'], 
                    self.states['margin'], 
                    self.states['model'], 
                    acc_min,
                    "convergency ",
                    self.dropout
                    )
            self.log("convergency training info: acc(%.3f) loss(%.5f)" % (acc, loss))
            if acc > 0: # set the min acc
                acc_min = 0.6 * acc

            if is_end: # finished train
                if acc < self.config['threshold_acc']: # not learned enough
                    self.updateStates({'lr': self.states['lr'] * self.config['gamma']})
                self.updateStates({"model": smodel})
            else:
                self.updateStates({'lr': self.states['lr'] * self.config["gamma"]})

            if loss < self.config['threshold_loss']:
                is_convergence = True

            # if still not convergenced, just make it convergenced
            if self.config['stop_lr'] > self.states['lr']:
                self.log("convergency: can't be convergenced!!!")
                break

            # make a test now
            self.testSteps(smodel)
        self.log("convergency finished.")
        return

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

        if self.solver == None:
            s_type = solver.split('.')[0].split('_')[-1]
            if s_type == "adadelta":
                self.solver = caffe.AdaDeltaSolver(solver)
            elif s_type == "rmsprop":
                self.solver = caffe.RMSPropSolver(solver)
            elif s_type == "adam":
                self.solver = caffe.AdamSolver(solver)
            elif s_type == "adagrad":
                self.solver = caffe.AdaGradSolver(solver)
            elif s_type == "nesterov":
                self.solver = caffe.NesterovSolver(solver)
            else:
                self.solver = caffe.SGDSolver(solver)
            self.log("used: "+s_type)
        self.solver_param = caffe.proto.caffe_pb2.SolverParameter()
        with open(solver, 'rt') as fd:
            pb2.text_format.Merge(fd.read(), self.solver_param)

        # restore the snapshot or model
        if model_snapshot:
            fsuffix = model_snapshot.split(".")
            m_name, s_name, crr_iter = self.modelName(model_snapshot)
            if os.path.exists(s_name):
                self.solver.restore(s_name)
            else:
                self.solver.net.copy_from(m_name)

        # init the batch size
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

