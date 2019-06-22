#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: Detecter.py
# Date: 2015 2015年05月21日 星期四 22时46分45秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.Mysql import Mysql
from lib.Util import *
from lib.MyThread import *
import numpy as np
import cv2
import cv

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Detecter(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        self.db = Mysql(conf["db_host"], conf["db_database"], conf["db_user"], conf["db_pass"])
    
    def test(self):
        if len(sys.argv) < 2:
            self.help()
        #print dir(cv2)
        face_cascade = cv2.CascadeClassifier(sys.argv[1])
        imgpath = sys.argv[2]
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)
        i = 0
        for (x,y,w,h) in faces:
            tmpimg = cv.LoadImage(imgpath)
            cv.SetImageROI(tmpimg, (x, y, w, int(h*1.15)))
            cv.SaveImage("face_"+str(i)+".jpg",tmpimg)
        return

    def help(self):
        print "Usage:", sys.argv[0], "cascade.xml pic.jpg"
        exit()

    def loadDatas(self):
        sql = "select pic_dir, fname, `md5` from images where status = '2' \
                and (type like 'star_%' or type like 'meinv') \
                limit 1000"
        res = self.db.query(sql)
        return res

    def finishPic(self, pic): # opencv
        face_cascade = cv2.CascadeClassifier(self.conf["cascade"])
        imgpath = self.conf['img_path'] + pic[0] + '/' + pic[1]
        img = cv2.imread(imgpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 7)
        i = 0
        faces_str = ""
        for (x,y,w,h) in faces:
            faces_str += pic[1] + ".face_"+str(i)+".jpg;"
            face_path = pic[3] + pic[1] + ".face_"+str(i)+".jpg"
            tmpimg = cv.LoadImage(imgpath)
            cv.SetImageROI(tmpimg, (x, y, w, int(h*1.15)))
            cv.SaveImage(face_path,tmpimg)
            print face_path
            i += 1
        db = Mysql(self.conf["db_host"], self.conf["db_database"], self.conf["db_user"], self.conf["db_pass"])
        sql = "update images set status='3', face_dir='%s' where `md5`='%s'" % (faces_str, pic[2])
        db.execute(sql)
        db.close()
        return

    def run(self):
        face_path = self.conf["img_path"] + 'face/'
        if not os.path.exists(face_path):
            os.mkdir(face_path)
        th = WorkerManager(self.conf["thread"])
        while True:
            ls = self.loadDatas()
            if len(ls) <= 0:
                break
            for one in ls:
                face_path = self.conf["img_path"] + 'face/' + one[0] + '/'
                if not os.path.exists(face_path):
                    os.mkdir(face_path)
                one.append(face_path)
                th.add_job(self.finishPic, [one])
        th.wait_for_complete()
        return

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
            "cascade": "../haarcascade_frontalface_default.xml",
    }
    t = Detecter(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

