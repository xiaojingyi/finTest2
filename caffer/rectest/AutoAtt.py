#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: AutoAtt.py
# Date: 2016 2016年03月09日 星期三 10时41分17秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
import numpy as np

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class AutoAtt(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: AutoAtt init")
        self.config = config
        self.debug = config["debug"]
        self.shape = config["shape"]
        self.shape_init = config['shape_init']
        self.att_dim = config["att_dim"]
        #super(AutoAtt, self).__init__(config)
    
    def testPrint(self):
        print "Hello World!"

    def genRandomAttrs(self, n, dim=100):
        mat = np.random.random((n, dim))
        return mat

    def checkAttrs(self, mat, last_eusorted=[]):
        len_mat = len(mat)
        eumax = 0
        eumin = 1000000
        euavg = 0
        print "---------checking---------"
        euls = []
        for i in range(len_mat):
            for j in range(len_mat):
                tmp = mat[i] - mat[j]
                eulen = np.dot(tmp, tmp)
                if i == 0:
                    euls.append(eulen)
                if eulen > 0:
                    euavg += eulen / (len_mat * (len_mat - 1))
                    if eulen > eumax:
                        eumax = eulen
                    elif eulen < eumin:
                        eumin = eulen
        eusorted = np.argsort(np.array(euls))
        print "max: ", eumax, 
        print "min: ", eumin, 
        print "avg: ", euavg, 
        if len(last_eusorted) > 0:
            tmp = np.array(last_eusorted) - eusorted
            print tmp
            ret = np.dot(tmp, tmp)
        else:
            ret = 0
        print ret
        return ret, eusorted

    def changeDim(self, crr_dim):
        if crr_dim == 0:
            return 1
        return 0

    def calcAttr(self, vec_re, att_mat):
        #print vec_re.shape, att_mat.shape
        len_att = len(att_mat)
        num = np.dot(vec_re, vec_re)
        vec_re = vec_re.reshape(len_att, 1)
        res = vec_re * att_mat
        tmp = np.zeros(att_mat.shape[1])
        for i in range(len_att):
            tmp += res[i]
        #print tmp
        tmp /= num
        #print res
        #print tmp
        #print num
        return tmp

    def brotherAttrMat(self, att_mat, re_mat):
        attrs_row = []
        for i in range(re_mat.shape[1]):
            attr_one = self.calcAttr(re_mat[:, i], att_mat)
            attrs_row.append(attr_one)
        attrs_row = np.array(attrs_row)
        #print attrs_row
        return attrs_row

    def run(self):
        self.init()
        pri_mat = self.genRandomAttrs(self.shape[self.shape_init], self.att_dim)
        flag_diff_a, sort_args_a = self.checkAttrs(pri_mat)
        print sort_args_a
        last_pri_mat = pri_mat
        #print pri_mat

        #print pri_mat.shape, self.mat.shape
        brother_mat = self.brotherAttrMat(pri_mat, self.mat)
        flag_diff_b, sort_args_b = self.checkAttrs(brother_mat)
        pri_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        flag_diff_a, sort_args_a = self.checkAttrs(pri_mat, sort_args_a)
        pri_mat += last_pri_mat
        pri_mat /= 2
        last_pri_mat = pri_mat

        while flag_diff_a or flag_diff_b:
            brother_mat = self.brotherAttrMat(pri_mat, self.mat)
            flag_diff_b, sort_args_b = self.checkAttrs(brother_mat, sort_args_b)
            pri_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
            flag_diff_a, sort_args_a = self.checkAttrs(pri_mat, sort_args_a)
            print flag_diff_a, flag_diff_b
            pri_mat += last_pri_mat
            pri_mat /= 2
            last_pri_mat = pri_mat
        print sort_args_b, sort_args_a

        """
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        brother_mat = self.brotherAttrMat(brother_mat, self.mat.transpose())
        brother_mat = self.brotherAttrMat(brother_mat, self.mat)
        """

    def init(self):
        fname = "./arr_mat.npy"
        if os.path.exists(fname):
            self.mat = np.load(fname)
        else:
            self.mat = np.random.randint(2, size=self.shape)
            np.save(fname, self.mat)
        print self.mat

    def reRow(self, index):
        return

    def reCol(self, index):
        return mat

    def src():

        return

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "shape": (100, 10),
            "shape_init": 0,
            "att_dim": 5000,
            }
    t = AutoAtt(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

