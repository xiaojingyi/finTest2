#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: FaceResizer.py
# Date: 2015 2015年06月03日 星期三 15时07分16秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
from lib.Util import *
from lib.MyThread import *
import cv, cv2

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class FaceResizer(object):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        if len(sys.argv) < 4:
            print sys.argv[0], "<size> <from> <to>"
            exit()

        self.resize = int(sys.argv[1])
        self.from_dir = sys.argv[2]
        self.to_dir = sys.argv[3]
    
    def mkdirs(self, dirs):
        crr = ""
        for one in dirs:
            crr += one + '/'
            if not os.path.exists(crr):
                try:
                    os.mkdir(crr)
                except:
                    1

    def doResize(self, img_path):
        t_img = img_path.replace(self.from_dir, self.to_dir)
        path_arr = t_img.split("/")
        fname = path_arr.pop()
        self.mkdirs(path_arr)
        tmpimg = cv2.imread(img_path)
        imgsize = list(tmpimg.shape[:2])
        print imgsize,
        newsize = imgsize[:]
        newsize[1] = self.resize
        newsize[0] = int(imgsize[0] * newsize[1] / imgsize[1])
        print newsize
        tmpimg = cv2.resize(tmpimg, (newsize[1], newsize[0])) 
        f_path = "/".join(path_arr) + '/' + fname
        print f_path
        cv2.imwrite(f_path,tmpimg)
        return

    def run(self):
        fls = walkDir(self.from_dir)
        th = WorkerManager(self.conf["thread"])
        for one in fls:
            f_img = one
            th.add_job(self.doResize, [f_img])
        th.wait_for_complete()
        return

    def testPrint(self):
        print "Hello World!"

def main():
    conf = {
            "thread": 10,
        }
    t = FaceResizer(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

