#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: ImgDownloader.py
# Date: 2015 2015年05月23日 星期六 20时00分47秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.Mysql import Mysql
from lib.Util import *
import cv, cv2
import threading

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

global sqls
global g_ct
sqls = []
g_ct = 0

class ImgDownloader(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        if not os.path.exists(conf["img_path"]):
            os.mkdir(conf["img_path"])
        self.db = Mysql(conf["db_host"], conf["db_database"], conf["db_user"], conf["db_pass"])
    
    def finish(self, one):
        global sqls
        global g_ct
        pid = one[0]
        pic_dir = self.conf["img_path"] + one[1]
        uri = one[2]
        fname = uri.split("/")[-1]
        if not os.path.exists(pic_dir):
            try:
                os.mkdir(pic_dir)
            except:
                1
        
        imgpath = pic_dir+"/"+fname
        cmd = "wget -t 3 -T 9 -q -O %s %s" % (imgpath,uri)
        print cmd
        is_break = False
        sql = ""
        try:
            exeCmd(cmd)
            tmpimg = cv.LoadImage(imgpath)
            sql = "update images set status = '1', fname='%s' where id = %s limit 1" % (fname, pid)
        except:
            sql = "update images set status = 'z' where id = %s limit 1" % (pid)
            os.remove(imgpath)

        sqls.append(sql)
        g_ct -= 1
        return

    def run(self):
        global sqls
        global g_ct
        i = 0
        p = self.conf["thread"] * 2
        ts = []
        while(True):
            if len(sqls) > 0:
                for j in range(len(sqls)):
                    sql = sqls.pop()
                    self.db.execute(sql)

            if g_ct > self.conf["thread"]:
                time.sleep(0.1)
                continue

            sql = "select id, pic_dir, uri from images where status='0' and \
                    pic_dir is not null order by id asc limit %s, %s" % (i, p)
            res = self.db.query(sql)
            if len(res) <= 0:
                print "finished"
                break

            for one in res:
                t=threading.Thread(target=self.finish, args=([one]))
                t.start()
                ts.append(t)
                g_ct += 1
            i += 1

        for one in ts:
            one.join()
        return

    def testPrint(self, i):
        print "Hello:", i

def main():
    conf = {
        "db_host": "localhost",
        "db_user": "root",
        "db_pass": "",
        "db_database": "baiduimg",
        "img_path": "/datas/images/",
        "thread": 10,
    }
    t = ImgDownloader(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

