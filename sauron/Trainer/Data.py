import os.path
#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-8-17 10:23:50$
# Note: This source file is NOT a freeware
# Version: Data.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-8-17 10:23:50$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
import numpy as np
import random
from lib.Util import *
import h5py
import skimage.transform
import json

class Data(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Data init")
        self.config = config
        if config.has_key("debug"):
            self.debug = config["debug"]   
        else:
            self.debug = False
        #super(Data, self).__init__(config)
        self.cp_st = 0
        self.mkDic(self.config['data_src'])
    
    def mkDic(self, src): # att data
        cache_f = "cache/person.cache"
        if os.path.exists(cache_f):
            tmp_str = getFileContent(cache_f, False)
            self.person = json.loads(tmp_str)
            return
        self.person = {}
        ls = os.listdir(src)
        index = 0
        for one in ls:
            one_dir = src + "/" + one
            if os.path.isdir(one_dir):
                id = one
                pics = walkDir(one_dir)
                person_imgs = []
                for one_pic in pics:
                    suffix = one_pic.split(".")[-1].lower()
                    if suffix in ["jpg", 'jpeg', 'png', 'pgm']:
                        person_imgs.append(one_pic)
                self.person[id] = {
                    "id": index,
                    "name": id,
                    "imgs": person_imgs,
                }
            index += 1
        tmp_str = json.dumps(self.person)
        writeToFile(cache_f, tmp_str)
        return 
    
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()
        
    def getPos(self, num):
        res = []
        for i in range(num):
            res.append(self.onePos())
        return res
    
    def getNag(self, num):
        res = []
        for i in range(num):
            res.append(self.oneNagtive())
        return res
    
    def onePos(self):
        keys = self.person.keys()
        key_len = len(keys)
        p1_i = random.randint(0, key_len-1)
        p1_k = keys[p1_i]
        p1 = self.person[p1_k]["imgs"]
        p1_len = len(p1)
        p1_img_i = random.randint(0, p1_len-1)
        p1_img = p1[p1_img_i]
        
        if p1_len == 1:
            p2_img_i = p1_img_i
        else:
            p2_img_i = random.randint(0, p1_len-2)
            if p2_img_i >= p1_img_i:
                p2_img_i += 1
        p2_img = p1[p2_img_i]
        return [p1_img, p2_img, '1', self.person[p1_k]['id']]
    
    def oneNagtive(self):
        keys = self.person.keys()
        key_len = len(keys)
        p1_i = random.randint(0, key_len-1)
        p1_k = keys[p1_i]
        p1 = self.person[p1_k]["imgs"]
        p1_img = p1[random.randint(0, len(p1)-1)]
        
        p2_i = random.randint(0, key_len-2)
        if p2_i >= p1_i:
            if p2_i+1 > key_len-1:
                p2_i = 0
            else:
                p2_i += 1
        p2_k = keys[p2_i]
        p2 = self.person[p2_k]["imgs"]
        p2_img = p2[random.randint(0, len(p2)-1)]
        return [p1_img, p2_img, '0', self.person[p1_k]['id']]
    
    def gen(self):
        num = self.config['db_size'] / 2
        data = []
        pos = self.getPos(num)
        data.extend(pos)
        nag = self.getNag(num)
        data.extend(nag)
        random.shuffle(data)
#        print np.array(data), len(data)
        self.mkDataSet(data)
    
    def mkDataSet(self, ls, set_name=False):
        if not set_name:
            set_name = self.config['train_dir_pre']
            
        one_size = self.config['width'] * self.config["height"] * 3
        with h5py.File(set_name, 'w') as f:
            Xset = f.create_dataset(
                'data', (1, 3, self.config['width'], self.config['height'],), 
                maxshape=(None, 3, self.config['width'], self.config['height'],), 
                dtype=np.float32, 
            )
            yset = f.create_dataset(
                'label', (1, ), 
                maxshape=(None, ), 
                dtype=np.float32, 
            )
            i = 0
            for one in ls:
                img_main = loadImgGray(one[0], False)
                img_main = skimage.transform.resize(img_main, (self.config['width'], self.config["height"]))
#                print img_main.shape
                final = np.zeros((3,self.config['width'], self.config["height"]))
#                print img_main.shape
                final[0, :, :] = img_main
                final[1, :, :] = 0
                final[2, :, :] = 0
                
                img_mate = loadImgGray(one[1], False)
                img_mate = skimage.transform.resize(img_mate, (self.config['width'], self.config["height"]))
#                print img_mate.shape
                final[2, :, :] = img_mate
                
                id = one[3]
                final[1, 0, 0] = one[2] # sim flag
                final[1, 0, 1] = one[4] # pair label
                final = np.array(final)
                
                label = id
                if i == 0:
                    Xset[0] = final.astype(np.float32)
                    yset[0] = np.array(label).astype(np.float32)
                else:
                    new_shape = list(Xset.shape)
                    new_shape[0] += 1
                    new_shape = tuple(new_shape)
                    Xset.resize(new_shape)
                    Xset[i] = final.astype(np.float32)
                    yset.resize((i+1, ))
                    yset[i] = np.array(label).astype(np.float32)
                i += 1
            print f['data'].shape, Xset.shape
            print f['label'].shape, yset.shape
        return
    
    def cp(self):
        self.cp_st = 0
        print "cp starting..."
        tmp_dir = self.config['train_dir_pre']
        tmp_test = self.config["test_db"]
        db_dir = self.config['train_dir']
        tmp = db_dir.split("/")
        del tmp[-1]
        db_dir = "/".join(tmp)
        cmd = "cp -r %s %s %s" % (tmp_dir, tmp_test, db_dir)
        exeCmd(cmd)
        cmd = "rm -fr %s %s " %  (tmp_dir, tmp_test)
        exeCmd(cmd)
        self.cp_st = 1
        return
    
    def run(self):
        tmp_dir = self.config['train_dir_pre']
        db_dir = self.config['train_dir']
        if not os.path.exists(tmp_dir):
            print "gen data..."
            self.gen()
        if not os.path.exists(db_dir):
            self.cp()        
        return
    
    def test(self):
        print "debug: ", self.debug
        
def main():
    config = {
        "debug": True,
    }
    model = Data(config)
    model.test()
    return

if __name__ == "__main__":
    main()
