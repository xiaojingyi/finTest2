#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: FaceSplit.py
# Date: 2015 2015年07月14日 星期二 20时11分25秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
import dlib
import numpy as np
from lib.Util import *
from lib.MyThread import *
import cv, cv2
from skimage import io

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class FaceSplit(object):
    landmark_conf = {
        "face": 16,
        "l_brow": 21,
        "r_brow": 26,
        "nose": 35,
        "l_eye": 41,
        "r_eye": 47,
        "mouse": 67,
    }
    
    def __init__(self, conf, debug=False):
        self.conf = {}
        self.conf = conf
        self.debug = debug
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(conf['model'])
    
    def dlibFace(self, imgfile):
        img = io.imread(imgfile)
        dets = self.detector(img, 1)
        face_info = []
        for k, d in enumerate(dets):
            pos = [d.left(), d.top(), d.right(), d.bottom()]
            shape = self.predictor(img, d)
            landmarks = []
            for i in range(68):
                tmp = shape.part(i)
                m_pos = [tmp.x, tmp.y]
#                landmarks.append(tmp)
                landmarks.append(m_pos)
#            print dir()
            # -TODO dlib alignment
            face_info.append({"pos": pos, "landmarks": landmarks, "img": img})
            
        return face_info
    
    def cvSplit(self, img_info):
        res = []
        for one in img_info:
            img = cv2.cvtColor(one['img'], cv2.COLOR_BGR2RGB)
            bbox = one['pos']
            landmarks = one['landmarks']
#            cv2.rectangle(img, tuple(bbox[0:2]), tuple(bbox[2:4]), (55,255,155), -1)
            boxes = {
                # element
                "head": [0, 0, 0, 0],
                "nose": [0, 0, 0, 0],
                "brow": [0, 0, 0, 0],
                "eyes": [0, 0, 0, 0],
                "mouse": [0, 0, 0, 0],
                # covered
                "face_right": [0, 0, 0, 0],
                "face_left": [0, 0, 0, 0],
                "face_top": [0, 0, 0, 0],
                "face_bottom": [0, 0, 0, 0],
                "face_middle": [0, 0, 0, 0],
                "face_t_right": [0, 0, 0, 0],
                "face_t_left": [0, 0, 0, 0],
                "face_b_right": [0, 0, 0, 0],
                "face_b_right": [0, 0, 0, 0],
            }
            landmarks_len = len(landmarks)
            x_center = y_center = 0
            y_max, x_max = img.shape[:2]
            for i in range(landmarks_len):
                l = landmarks[i]
                x = l[0]
                y = l[1]
                if i == 8:
                    x_center = x
                if i == 30:
                    y_center = y
                if i <= self.landmark_conf['face']:
                    boxes["head"] = self.boxAsign(boxes["head"], x, y)
                elif i <= self.landmark_conf['r_brow']:
                    boxes["brow"] = self.boxAsign(boxes["brow"], x, y)
                elif i <= self.landmark_conf['nose']:
                    boxes["nose"] = self.boxAsign(boxes["nose"], x, y)
                elif i <= self.landmark_conf['r_eye']:
                    boxes["eyes"] = self.boxAsign(boxes["eyes"], x, y)
                else:
                    boxes["mouse"] = self.boxAsign(boxes["mouse"], x, y)
#                cv2.circle(img, tuple(l), 3, (0,0,255),-1) 
#                cv2.imshow("image", img)
#                cv2.waitKey(0)
            imgs = self.boxesImg(img, boxes, x_center, y_center, x_max, y_max)
            res.append(imgs)
#            bbox = boxes["eyes"]
#            cv2.rectangle(img, tuple(bbox[0:2]), tuple(bbox[2:4]), (0,0,0), -1)
#            crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#            cv2.imshow("image", crop_img)
#            res=cv2.resize(crop_img,(32,32),interpolation=cv2.INTER_CUBIC, fx=x_max, fy=y_max)
#            cv2.imshow('iker',res)
#            cv2.imshow("primary", img)
        return res
    
    def boxesImg(self, img, boxes, x_center, y_center, x_max, y_max):
        imgs = {}
        # head pos
        len1 = boxes['head'][2] - x_center
        len2 = x_center - boxes['head'][0]
        if len1 > len2:
            boxes['head'][0] -= (len1 - len2)
            if boxes['head'][0] < 0:
                boxes['head'][0] = 0
        else:
            boxes['head'][2] += (len2 - len1)
            if boxes['head'][2] > x_max:
                boxes['head'][2] = x_max
        len1 = boxes['head'][3] - y_center
        len2 = y_center - boxes['head'][1]
        if len1 > len2:
            boxes['head'][1] -= (len1 * 2 - len2)
            if boxes['head'][1] < 0:
                boxes['head'][1] = 0
        else:
            boxes['head'][3] += (len2 - len1)
            if boxes['head'][3] > y_max:
                boxes['head'][3] = y_max
        w = boxes['head'][2] - boxes['head'][0]
        y = boxes['head'][3] - boxes['head'][1]
        if w > y:
            1
        else:
            l_pad = int((y - w) / 2)
            r_pad = y - w - l_pad
            boxes['head'][0] -= l_pad
            if boxes['head'][0] < 0:
                boxes['head'][0] = 0
            boxes['head'][2] += r_pad
            if boxes['head'][3] > y_max:
                boxes['head'][3] = y_max
        bbox = boxes['head']
#        print bbox
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#        cv2.imshow("image", crop_img)
        face_img = cv2.resize(crop_img,(227,227),interpolation=cv2.INTER_CUBIC)
        imgs['head'] = face_img
        self.show(face_img)
        
        imgs['head_gray'] = gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        self.show(gray_img)
        
        tmp_img = gray_img.copy();
        cv2.rectangle(tmp_img, (113, 0), (227, 227), (0,0,0), -1)
        imgs['face_left'] = tmp_img
        self.show(tmp_img)
        
        tmp_img = gray_img.copy();
        cv2.rectangle(tmp_img, (0, 0), (113, 227), (0,0,0), -1)
        imgs['face_right'] = tmp_img
        self.show(tmp_img)

        tmp_img = gray_img.copy();
        cv2.rectangle(tmp_img, (0, 140), (227, 227), (0,0,0), -1)
        imgs['face_top'] = tmp_img
        self.show(tmp_img)

        tmp_img = gray_img.copy();
        cv2.rectangle(tmp_img, (0, 0), (227, 140), (0,0,0), -1)
        imgs['face_bottom'] = tmp_img
        self.show(tmp_img)

        tmp_img = gray_img.copy();
        cv2.rectangle(tmp_img, (0, 0), (227, 70), (0,0,0), -1)
        cv2.rectangle(tmp_img, (0, 150), (227, 227), (0,0,0), -1)
        imgs['face_middle'] = tmp_img
        self.show(tmp_img)
        
        tmp_img = gray_img.copy();
        cv2.fillConvexPoly(tmp_img, np.array([[0,227], [227,0], [227,227]]), 1)
        imgs['face_t_left'] = tmp_img
        self.show(tmp_img)
        
        tmp_img = gray_img.copy();
        cv2.fillConvexPoly(tmp_img, np.array([[0,0], [0,227], [227,227]]), 1)
        imgs['face_t_right'] = tmp_img
        self.show(tmp_img)
        
        tmp_img = gray_img.copy();
        cv2.fillConvexPoly(tmp_img, np.array([[0,0], [227,227], [227,0]]), 1)
        imgs['face_b_left'] = tmp_img
        self.show(tmp_img)
        
        tmp_img = gray_img.copy();
        cv2.fillConvexPoly(tmp_img, np.array([[0,0], [0,227], [227,0]]), 1)
        imgs['face_b_right'] = tmp_img
        self.show(tmp_img)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs['nose'] = self.cutArea(boxes['nose'], gray_img, x_max, y_max)
        self.show(imgs['nose'])
        imgs['mouse'] = self.cutArea(boxes['mouse'], gray_img, x_max, y_max)
        self.show(imgs['mouse'])
        imgs['brow'] = self.cutArea(boxes['brow'], gray_img, x_max, y_max)
        self.show(imgs['brow'])
        imgs['eyes'] = self.cutArea(boxes['eyes'], gray_img, x_max, y_max)
        self.show(imgs['eyes'])

        return imgs
    
    def cutArea(self, bbox, img, x_max, y_max):
        print bbox, x_max, y_max
        l1 = bbox[2] - bbox[0]
        l2 = bbox[3] - bbox[1]
        pad = int(abs(l1 - l2) / 2)
        if l1 > l2:
            bbox[3] += pad
            if bbox[3] > y_max:
                bbox[3] = y_max
            bbox[1] -= pad
            if bbox[1] < 0:
                bbox[1] = 0
        else:
            bbox[2] += pad
            if bbox[2] > x_max:
                bbox[2] = x_max
            bbox[0] -= pad
            if bbox[0] < 0:
                bbox[0] = 0
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        try:
            face_img = cv2.resize(crop_img,(227,227),interpolation=cv2.INTER_CUBIC)
        except:
            print "err: ", bbox
        return face_img
    
    def show(self, img):
        if self.debug:
            cv2.imshow('iker',img)
            cv2.waitKey(0)
        return
    
    def boxAsign(self, box, x, y):
        if box[0] == 0 or box[0] > x: # left
            box[0] = x
        if box[1] == 0 or box[1] > y: # top
            box[1] = y
        if box[2] < x: # right
            box[2] = x
        if box[3] < y: # bottom
            box[3] = y
        return box
    
    def topAsign(self, top, y):
        if top == 0 or top > y:
            return y
        return top
    
    def bottomAsign(self, bottom, y):
        if bottom == 0 or bottom > y:
            return y
        return bottom
    
    def oneImg(self, img_path):
        file_label = img_path.split(" ")
        fname = file_label[0]
        label = file_label[1]
        prev_fname = fname
        
        suffix = fname.split(".")[-1]
        prefix = fname.split("/")[0]
        
        res = self.dlibFace(prev_fname)
        faces_final = self.cvSplit(res)
        len_faces = len(faces_final)
        for i in range(len_faces):
            obj = faces_final[i]
            for key in obj.keys():
                fname = prefix + "_" + key + "/" + prev_fname + (".%d.%s." % (i, label)) + suffix
                dir_ls = fname.split("/")[0:-1]
                mkdirs(dir_ls)
                cv2.imwrite(fname, obj[key])
#        print dir_ls
#        exit()
            
    def run(self, img_ls_file, root_dir):
        ls = readFileLines(img_ls_file)
        wm = WorkerManager(15)
        for one in ls:
#            self.oneImg(root_dir + "/" + one)
            fpath = root_dir + "/" + one
            wm.add_job(self.oneImg, [fpath])
        wm.wait_for_complete()
        
    def testPrint(self):
        res = self.dlibFace(sys.argv[1])
        print res
        res = self.cvSplit(res)
        print res
        print "Hello World!"

def main():
    conf = {
        "model": "/datas/pkgs/shape_predictor_68_face_landmarks.dat",
    }
    t = FaceSplit(conf, False)
    t.run(sys.argv[1], sys.argv[2])
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

