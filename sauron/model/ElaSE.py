#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-7-20 10:57:06$
# Note: This source file is NOT a freeware
# Version: ElaSE.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-7-20 10:57:06$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("/datas/lib/py")
from lib.Util import *
import numpy as np
from CaffeFeature import *
from elasticsearch import Elasticsearch

class ElaSE(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: ElaSE init")
        self.config = config
        self.debug = config["debug"]
        #super(ElaSE, self).__init__(config)
        self.loadModels()
        self.es = Elasticsearch(config["nodes"])
        
    def indexDir(self, dir_path):
        ls = walkDir(dir_path)
        i = 0
        for one in ls:
            print i, one
            doc = self.mkElaData(one)
            res = self.es.index(index="face", doc_type="txt", body=doc)
            i += 1
            
    def search(self, img_path):
        doc = self.mkElaData(img_path)
        q = {
            "query": {
                "match" : {
                    "head" : {
                        "query": doc['head'],
                        "operator" : "or",
                    }
                }
            },
            "_source": ["head", "head_f", "img_path"], 
            "from": 0,
            "size": self.config["search_limit"], 
        }

        res = self.es.search(index="face", body=q, timeout=10000)
#        print doc['head_f'].split(",")
        q_feature = np.array(doc['head_f'].split(","), np.float32)
        print len(q_feature), q_feature
        for i in range(len(res['hits']["hits"])):
            one = res['hits']["hits"][i]["_source"]
            t_feature = np.array(one['head_f'].strip(",").split(","), np.float32)
            similar = cosSimilar(q_feature, t_feature)
            similar2 = pearsonSimilar(q_feature, t_feature)
            similar3 = euclidSimilar(q_feature, t_feature)
            res['hits']["hits"][i]['similar'] = ( similar + similar2 + similar3 ) / 3
#            res['hits']["hits"][i]['similar'] =similar3
        res['hits']["hits"] = sorted(res['hits']["hits"], key=lambda s: s['similar'], reverse = True)
        return res
        
    def mkElaData(self, img_path):
        features = {
            "img_path": img_path
        }
        for one in self.models:
            features[one["name"]] = ""
            features[one["name"]+"_f"] = ""
            tmp = list(one['model'].process(img_path, "deepid_encode512"))
#            print tmp
#            tmp.extend(list(one['model'].levelData("deepid_1")))
#            print one['model'].levelData("deepid_1")
#            tmp.extend(list(one['model'].levelData("deepid_2")))
#            print one['model'].levelData("deepid_2")
            tmp = np.array(tmp)
#            print len(tmp), tmp
#            exit()
            for i in range(len(tmp)):
                tmp[i] += 1
                bin_h = round(tmp[i])
                features[one["name"]] += "%s_%d_%d " % (one['name'], i, bin_h)
            features[one["name"]] = features[one["name"]].strip()
#            tmp = one['model'].process(img_path, "fc9")
#            print len(tmp), tmp
            for i in range(len(tmp)):
                features[one["name"]+"_f"] += "%f," % tmp[i]
            features[one["name"]+"_f"] = features[one["name"]+"_f"].strip(",")
#            print len(features[one["name"]].split(","))
#        print features
#            print features[one["name"]].split(",")
#            print tmp
#            print len(tmp)
        return features
            
    def loadModels(self):
        self.models = []
        for one in self.config['models']:
            conf = {
                "name": one['name'],
                "model": one['model'],
                "mean": one['mean'],
                "deploy": one['deploy'],
                "gpu": self.config['gpu'],
            }
            tmp = CaffeFeature(conf)
            self.models.append({"name": one['name'], "model": tmp})
        return
    
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()
        
    def test(self):
#        self.mkElaData(sys.argv[1])
#        self.indexDir(sys.argv[1])
        res = self.search(sys.argv[1])
        for one_hit in  res['hits']["hits"]:
            print one_hit["_source"]["img_path"]
#            print one_hit["_source"]["head"]
            print one_hit["_score"]
            print one_hit["similar"]
            
        print "debug: ", self.debug
        
def main():
    config = {
        "debug": True,
        
        # caffe
        "gpu": True,
        "models":[
            {
                "name": "head",
                "deploy": "/datas/root/deepid_test/models/sauron/sauron_google_deploy.prototxt",
                "model": "/datas/root/deepid_test/models/sauron/sauron_google_iter_780000.caffemodel",
                "mean": "/datas/root/deepid_test/dataset/att_train_mean.binaryproto",
            },
        ],
        
        # elastic search
        "nodes": ["http://localhost:9200"],
        "search_limit": 100,
    }
    model = ElaSE(config)
    model.test()
    return

if __name__ == "__main__":
    main()
