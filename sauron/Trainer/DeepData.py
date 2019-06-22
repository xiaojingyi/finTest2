#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-9-11 17:30:20$
# Note: This source file is NOT a freeware
# Version: DeepData.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-9-11 17:30:20$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
import numpy as np
import random
from lib.Util import *
from Data import Data

class DeepData(Data):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DeepData init")
        self.config = config
        if config.has_key("debug"):
            self.debug = config["debug"]   
        else:
            self.debug = False
        self.index = [0, 0]
        self.data = []
        super(DeepData, self).__init__(config)
    
    def getSame(self, index):
        keys = self.person.keys()
        person = self.person[keys[index[0]]]['imgs']
        len_person = len(person)
        choose = random.randint(0, len_person-1)
        return person[choose], self.person[keys[index[0]]]['id']
    
    def getDiff(self, index):
        keys = self.person.keys()
        key_len = len(keys)
        choose = random.randint(0, key_len-2)
        if index[0] < key_len - 1: # not the last one
            if choose >= index[0]:
                choose += 1
        person = self.person[keys[choose]]['imgs']
        len_person = len(person)
        choose = random.randint(0, len_person-1)
        return person[choose], self.person[keys[choose]]['id']
    
    def mkTest(self, test_dir):
        keys = self.person.keys()
        names = os.listdir(test_dir)
        index = [0, 0]
        data = []
        for one in names:
            person = self.person[one]
            index[0] = keys.index(one)
            pics = walkDir(test_dir+"/"+one)
            for one_pic in pics:
                choose = random.randint(1, 1000)
                index[1] = random.randint(0, len(person['imgs']) - 1)
                sim = '0'
                if choose > 500: # get the same
                    tmp = self.getSame(index)
                    pair_pic = tmp[0]
                    pair_id = tmp[1]
                    sim = '1'
                else:
                    tmp = self.getDiff(index)
                    pair_pic = tmp[0]
                    pair_id = tmp[1]
                data.append([one_pic, pair_pic, sim, person['id'], pair_id])
        return data
    
    def mkData(self, size_all, size_index):
        assert size_all > size_index, "size error"
        
        if len(self.data) == 0:
            self.data = self.iters(size_all)
        else:
            for i in range(size_index):
                len_data = len(self.data)
                choose = random.randint(0, len_data - 1)
                del self.data[choose]
            tmp = self.iters(size_index)
            self.data.extend(tmp)
            
        assert len(self.data) == size_all, "data error"
        return self.data
    
    def iters(self, size):
        keys = self.person.keys()
        key_len = len(keys)
        data = []
        for i in range(size):
            # get the current iteration person
            person = self.person[ keys[ self.index[0] ] ]
            len_person = len(person['imgs'])
            person_pic = person['imgs'][ self.index[1] ]
            
            # get the pair
            t = random.randint(1, 1000)
            pair_pic = False
            sim = '0'
            label = person['id']
            if t > 500:
                tmp = self.getSame(self.index)
                pair_pic = tmp[0]
                pair_id = tmp[1]
                sim = '1'
            else:
                tmp = self.getDiff(self.index)
                pair_pic = tmp[0]
                pair_id = tmp[1]
                
            # load datas
            data.append([person_pic, pair_pic, sim, label, pair_id])
                
            # update the self.index
            self.index[1] += 1
            if self.index[1] >= len_person: # end of one person
                self.index[0] += 1
                if self.index[0] >= key_len: # at the last one
                    self.index[0] = 0
                self.index[1] = 0
        return data
    
    def gen(self):
        index_f = "cache/index.npy"
        if os.path.exists(index_f):
            self.index = np.load(index_f)
        data = self.mkData(self.config['db_size'], self.config['iter_size'])
        test = self.mkTest(self.config["test_src"])
        print "data iter: ", self.index
        random.shuffle(data)
        random.shuffle(test)
        self.mkDataSet(data)
        self.mkDataSet(test, self.config["test_db"])
        np.save(index_f, self.index)
        
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()
        
    def test(self):
        print "debug: ", self.debug
        
def main():
    config = {
        "debug": True,
    }
    model = DeepData(config)
    model.test()
    return

if __name__ == "__main__":
    main()
