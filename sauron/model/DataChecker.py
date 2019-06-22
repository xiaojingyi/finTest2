#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DataChecker.py
# Date: 2015 2015年05月23日 星期六 18时53分17秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.Mysql import Mysql
from lib.Util import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DataChecker(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        self.db = Mysql(conf["db_host"], conf["db_database"], conf["db_user"], conf["db_pass"])
        self.rc = []
    
    def getOneRecord(self):
        if len(self.rc) <= 0:
            sql = "select spec, count(*), pic_dir  from images where pic_dir is null and (type = 'star_female' or type = 'star_male') group by spec order by spec limit 50"
            self.rc = self.db.query(sql)
        if len(self.rc) <= 0:
            print "empty"
            exit()
        one = self.rc.pop()
        print one[0], one[1], one[2]
        return one[0]

    def run(self):
        spec = ""
        while(True):
            spec = self.getOneRecord()
            cmd = raw_input()
            sql = ""
            if cmd == "exit":
                break
            else:
                sql = "update images set pic_dir = '%s' where pic_dir is null and spec = '%s'" % (cmd, spec)

            self.db.execute(sql)
            print sql
        return

    def testPrint(self):
        print "Hello World!"

def main():
    conf = {
            "db_host": "localhost",
            "db_user": "root",
            "db_pass": "",
            "db_database": "baiduimg",
    }
    t = DataChecker(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

