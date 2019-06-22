#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-5-4 7:25:14$
# Note: This source file is NOT a freeware
# Version: Test.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-5-4 7:25:14$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
import numpy as np
import cv2
from matplotlib import pyplot as plt

class Test(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: Test init")
        self.config = config
        self.debug = config["debug"]
        #super(Test, self).__init__(config)
        
    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()
        
    def videoCapture(self):
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        
    def line(self):
        # Create a black image
        img = np.zeros((512,512,3), np.uint8)

        # Draw a diagonal blue line with thickness of 5 px
        imgret = cv2.line(img,(0,0),(511,511),(255,0,0),5)
        imgret = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
        imgret = cv2.circle(img,(447,63), 63, (0,0,255), -1)
        imgret = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
        pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
        pts = pts.reshape((-1,1,2))
        imgretret = cv2.polylines(img,[pts],True,(0,255,255))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2)
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        
    def showEvents(self):
        import cv2
        events = [i for i in dir(cv2) if 'EVENT' in i]
        print events
        
    def mouse(self):
        global ix,iy,drawing,mode
        drawing = False # true if mouse is pressed
        mode = True # if True, draw rectangle. Press 'm' to toggle to curve
        ix,iy = -1,-1

        # mouse callback function
        def draw_circle(event,x,y,flags,param):
            global ix,iy,drawing,mode
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix,iy = x,y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing == True:
                    if mode == True:
                        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
                    else:
                        cv2.circle(img,(x,y),5,(0,0,255),-1)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if mode == True:
                    cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
                else:
                    cv2.circle(img,(x,y),5,(0,0,255),-1)
        img = np.zeros((512,512,3), np.uint8)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle)

        while(1):
            cv2.imshow('image',img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                mode = not mode
            elif k == 27: # esc
                break

        cv2.destroyAllWindows()
        
    def test(self):
#        self.videoCapture()
#        self.line()
#        self.showEvents()
        self.mouse()
        
def main():
    config = {
        "debug": True,
    }
    model = Test(config)
    model.test()
    return

if __name__ == "__main__":
    main()
