#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: ImgSet.py
# Date: 2015 2015年05月23日 星期六 08时38分02秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.Mysql import Mysql
from lib.Util import *
import urllib

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
reload(sys)
sys.setdefaultencoding('utf8')

class ImgSet(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        self.db = Mysql(conf["db_host"], conf["db_database"], conf["db_user"], conf["db_pass"])
    
    def loadsData(self, data):
        print "loadsData should be rewrite!"
        return data

    def uriTemplate(self, uri_tmp, uri_referer, data_key):
        self.uri = uri_tmp
        self.referer = uri_referer
        self.data_key = data_key
        return

    def setUriParam(self, data):
        self.s = data
        for k in data.keys():
            self.uri = self.uri.replace("<%s>"%(k), urllib.quote(data[k]))
            self.referer = self.referer.replace("<%s>"%(k), urllib.quote(data[k]))
        return

    def fetchSet(self, start=0, limit=60):
        res = {}
        uri = self.uri.replace("<start>", str(start))
        uri = uri.replace("<limit>", str(limit))
        #print uri
        #print self.referer
        text = fetchUrl(uri, param={"referer":self.referer})
        res = self.loadsData(text)
        return res

    def save(self, data, loop_i):
        sql = "select `md5`, tags from images where `md5` = '%s' limit 1" % (data["md5"])
        res = self.db.query(sql)
        #print res
        tag = data["tags"]
        if len(res) > 0: # update
            tags = res[0][1].split(",")
            tags_dic = {}
            for t in tags:
                tags_dic[md5(t)] = 1
            if not tags_dic.has_key(md5(tag)):
                sql = "update images set tags = concat(tags, ',%s') where `md5` = '%s'" % (tag, data["md5"])
                #print sql
                self.db.execute(sql)
        else: # insert
            keys = ""
            values = ""
            for k in data.keys():
                keys += " `%s`, " % (k)
                values += " '%s' ," % (data[k])
            keys = keys.strip(" ,") + ", createtime, fetch_loop "
            values = values.strip(" ,") + ", now(), %s " % (str(loop_i))
            sql = "insert into images ( %s ) values( %s )" % (keys, values)
            self.db.execute(sql)
            #print keys, values
            #print sql
        #exit()
        return

    def testPrint(self):
        print "Hello World!"

def main():
    conf = {}
    t = ImgSet(conf)
    t.testPrint()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

