#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-9-2 21:36:14$
# Note: This source file is NOT a freeware
# Version: DatasetTrainVal.py 0.1 jingyi Exp $
#
# NOTE: ONLY FOR ATT STYLE DATASETS!!!

__author__="jingyi"
__date__ ="$2015-9-2 21:36:14$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
import numpy as np
import random
from lib.Util import *

class DatasetTrainVal(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DatasetTrainVal init")
        self.config = config
        self.debug = config["debug"]
        #super(DatasetTrainVal, self).__init__(config)
        
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()
    
    def split(self, dataset, per):
        dataset = dataset.strip("/")
        print "spliting", dataset, "..."
        dir_ls = os.listdir(dataset)
        len_dir = len(dir_ls)
        for i in range(len_dir):
            one = dataset + '/' + dir_ls[i]
            ls = walkDir(one)
            len_one = len(ls)
            random.shuffle(ls)
            for j in range(len_one):
                from_path = ls[j]
                suffix = from_path.split(".")[-1].lower()
                if suffix not in ["jpg", 'jpeg', 'png', 'pgm']:
                    continue
                if j < int(len_one * per):
                    to_path = dataset+"_val/"+from_path
                    cpWithPath(from_path, to_path)
                else:
                    to_path = dataset+"_train/"+from_path
                    cpWithPath(from_path, to_path)
            print "processed: ", dir_ls[i]
            
    def run(self):
        try:
            set_dir = sys.argv[1]
            split_per = sys.argv[2]
        except:
            print "note: only for att style datasets!"
            print "usage:", sys.argv[0], "dataset", "validate_per"
            exit()
        self.split(set_dir, float(split_per))
        
    def test(self):
        print "debug: ", self.debug
        
def main():
    config = {
        "debug": True,
    }
    model = DatasetTrainVal(config)
    model.run()
    return

if __name__ == "__main__":
    main()
