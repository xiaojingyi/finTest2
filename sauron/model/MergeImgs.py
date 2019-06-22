#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-7-22 17:42:02$
# Note: This source file is NOT a freeware
# Version: MergeImgs.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-7-22 17:42:02$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
from lib.Util import *
from lib.MyThread import *
import numpy as np
import skimage
import skimage.io
import skimage.color
import random
import math

global wm
wm = WorkerManager(5)

class MergeImgs(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: MergImgs init")
        self.config = config
        self.debug = config["debug"]
        #super(MergImgs, self).__init__(config)
    
    def mkDic(self, ls): # att data
        self.person = {}
        for one in ls:
            tmp = one.split("/")
            id = tmp[-2]
            if self.person.has_key(id):
                self.person[id].append(one)
            else:
                self.person[id] = [one]
                
    def posDatas(self):
        train = []
        test = []
        data = []
        keys = self.person.keys()
        for key in keys:
            for img1 in self.person[key]:
                for img2 in self.person[key]:
                    data.append([img1, img2, 1])
        random.shuffle(data)
        len_data = len(data)
        for i in range(len_data):
            if i < 0.2 * len_data:
                test.append(data[i])
            else:
                train.append(data[i])
        return train, test
    
    def oneNagtive(self):
        keys = self.person.keys()
        key_len = len(keys)
        p1_i = random.randint(0, key_len-1)
        p1_k = keys[p1_i]
        p1 = self.person[p1_k]
        p1_img = p1[random.randint(0, len(p1)-1)]
        
        p2_i = random.randint(0, key_len-2)
        if p2_i >= p1_i:
            p2_i += 1
        p2_k = keys[p2_i]
        p2 = self.person[p2_k]
        p2_img = p2[random.randint(0, len(p2)-1)]
        return [p1_img, p2_img, 0]

    def nagtivDatas(self, len_train, len_test):
        train = []
        test = []
        for i in range(len_train):
            train.append(self.oneNagtive())
        for i in range(len_test):
            test.append(self.oneNagtive())
        return train, test
    
    def mkDatas(self, source, target):
        global wm
        ls = walkDir(source)
        self.mkDic(ls)
#        data = []
        train = []
        test = []
        # positive
        tr, t = self.posDatas()
        for i in range(1):
            train.extend(tr)
        test.extend(t)
        print len(train), len(test)
        print np.array(train)
        print np.array(test)

        tr, t = self.nagtivDatas(len(train), len(test)*3)
        train.extend(tr)
        test.extend(t)
        print len(train), len(test)
        print np.array(train)
        print np.array(test)

        # write datasets
        for i in range(len(train)):
            one = train[i]
            src_path = one[0]
            to_path = source + "_train/" + src_path.split("/")[-2] + "/%d.%d.png" %(i, one[2])
#            self.mergeOne(one[0], one[1], to_path)
            param = [one[0], one[1], to_path]
            wm.add_job(self.mergeOne, [param])
        wm.wait_for_complete()
        for i in range(len(test)):
            one = test[i]
            src_path = one[0]
            to_path = source + "_val/" + src_path.split("/")[-2] + "/%d.%d.png" %(i, one[2])
#            self.mergeOne(one[0], one[1], to_path)
            param = [one[0], one[1], to_path]
            wm.add_job(self.mergeOne, [param])
        wm.wait_for_complete()
                
    def mergeOne(self, param):
        img1_path = param[0]
        img2_path = param[1]
        to = param[2]
        img1 = skimage.img_as_float(skimage.io.imread(img1_path)).astype(np.float32)
        img2 = skimage.img_as_float(skimage.io.imread(img2_path)).astype(np.float32)
        img1_g = skimage.color.rgb2gray(img1)
        img2_g = skimage.color.rgb2gray(img2)
#        print img1_g.shape
#        print img2_g.shape
        final = img1_g[:, :, np.newaxis]
        final = np.tile(final, (1, 1, 3))
#        print final.shape
#        print final[:, :, 0].shape, img2_g[:, :].shape
        final[:, :, 1] = 0
        final[:, :, 2] = img2_g[:, :]
#        print final
        to_dir = to.split("/")[:-1]
        mkdirs(to_dir)
        skimage.io.imsave(to, final)
#        tmp = skimage.img_as_float(skimage.io.imread(to)).astype(np.float32)
#        print tmp == final
        
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()
        
    def test(self):
#        self.mergeOne(sys.argv[1], sys.argv[2], sys.argv[3])
        self.mkDatas(sys.argv[1], sys.argv[2])
        print "debug: ", self.debug
        
def main():
    config = {
        "debug": True,
    }
    model = MergeImgs(config)
    model.test()
    return

if __name__ == "__main__":
    main()
