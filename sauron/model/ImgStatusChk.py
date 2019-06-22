#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: ImgStatusChk.py
# Date: 2015 2015年05月26日 星期二 13时14分29秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.Mysql import Mysql
from lib.Util import *
import cv, cv2

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class ImgStatusChk(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        self.db = Mysql(conf["db_host"], conf["db_database"], conf["db_user"], conf["db_pass"])

    def run(self):
        sql = "select pic_dir, fname, `md5` from images where status = '1' "
        res = self.db.query(sql)
        i = 0
        for one in res:
            i += 1
            imgpath = ""
            try:
                imgpath = self.conf["img_path"] + one[0] + "/" + one[1]
                tmpimg = cv.LoadImage(imgpath)
                print i, cv.GetSize(tmpimg)
                sql = "update images set status = '2' where `md5` = '%s'" % (one[2])
                self.db.execute(sql)
            except:
                sql = "update images set status = 'z' where `md5` = '%s'" % (one[2])
                print sql
                self.db.execute(sql)
                try:
                    if not imgpath:
                        os.remove(imgpath)
                except:
                    1

    def testPrint(self):
        print "Hello World!"

def main():
    conf = {
        "db_host": "localhost",
        "db_user": "root",
        "db_pass": "",
        "db_database": "baiduimg",
        "img_path": "/datas/images/",
        "thread": 10,
    }
    t = ImgStatusChk(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

