#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: IndexCreator.py
# Date: 2015 2015年06月01日 星期一 21时46分05秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

from lib.Mysql import Mysql
from lib.Util import *
from lib.MyThread import *
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch

class IndexCreator(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        self.es = Elasticsearch(conf["nodes"], timeout=10000)
        self.db = Mysql(conf["db_host"], conf["db_database"], conf["db_user"], conf["db_pass"])
    
    def groovyCode(self):
        print "groovy code is: "
        code = "-("
        for i in range(self.conf["item_len"]):
            code += "abs(doc['f%d'].value-f%d) + " % (i, i)
        code += "0)"
        print code
        writeToFile(self.conf["groovy_path"]+self.conf['groovy_file'], code)
        return

    def indexData(self, data):
        index_data = {}
        for i in range(len(data)):
            index_data["f"+str(i)] = int(data[i])
        return index_data

    def loadDatas(self):
        sql = "select `md5`, face_dir, pic_dir, uri from images where status = '3' limit 1000"
        res = self.db.query(sql)
        return res

    def exactor(self, one):
        faces = one[1].split(';')
        face_dir = one[-1]
        for face in faces:
            all_finished = True
            if face:
                face_path = face_dir + one[2] + '/' + face
                cmd = "../cmodel/featurebr/build/feature %s %s " % ("FaceRecognition", face_path)
                print cmd
                res = exeCmd(cmd)
                data = res.strip(',').split(',')
                ldata = len(data)
                if ldata != self.conf['item_len']:
                    print "len of data error: %d" % ldata
                    all_finished = False
                    continue
                data = self.indexData(data)
                data['hash_id'] = one[0]
                data['path'] = face_path
                data['uri'] = one[3]
                res = self.es.index(index="face", doc_type="feature", body=data)
            if all_finished:
                db = Mysql(self.conf["db_host"], self.conf["db_database"], self.conf["db_user"], self.conf["db_pass"])
                sql = "update images set status = '4' where `md5` = '%s' " % one[0]
                db.execute(sql)
                db.close()
        return

    def run(self):
        self.es.indices.delete("face")
        res = self.es.indices.create(index="face")
        print res
        self.es.indices.flush("face")

        face_dir = self.conf["img_path"] + 'face/'
        th = WorkerManager(self.conf["thread"])
        while True:
            ls = self.loadDatas()
            if len(ls) <= 0:
                break
            for one in ls:
                one.append(face_dir)
#                self.exactor(one)
                th.add_job(self.exactor, [one])
        th.wait_for_complete()
        return

    def test(self):
        cmd = '../cmodel/featurebr/build/feature ' + sys.argv[1]
        res = exeCmd(cmd)
        data = res.strip(',').split(',')
        ldata = len(data)
        if ldata != self.conf['item_len']:
            print "len of data error: %d" % ldata
            exit()
        data = self.indexData(data)
        print data

        q = { "query": 
                { "function_score": { 
                    "functions": [ { "script_score": {
                        "params": data, 
                        "script": "distance_face"
                        } } ], 
                    "query": { "match_all": {} }, 
                    "score_mode": "first" 
                    } 
                }, 
                "_source": ["uri", "path", "hash_id"], 
                "from": 0, 
                "size": self.conf["search_limit"], 
            } 

        res = self.es.search(index="face", body=q, timeout=100000) 
        printJson (res)
        return res

    def testPrint(self):
        print "Hello World!"

def main():
    conf = {
            "db_host": "localhost",
            "db_user": "root",
            "db_pass": "",
            "db_database": "baiduimg",
            "img_path": "/datas/images/",
            "thread": 8,
            "nodes": ["http://localhost:9200"],
            "item_len": 768,
            "groovy_path": "/datas/root/elasticsearch-1.5.2/config/scripts/",
            "groovy_file": "distance_face.groovy",
            "search_limit": 50,
        }
    t = IndexCreator(conf)
    t.test()
#    t.groovyCode()
#    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

