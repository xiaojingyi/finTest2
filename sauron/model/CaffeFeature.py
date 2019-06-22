#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: CaffeFeature.py
# Date: 2015 2015年07月13日 星期一 19时26分27秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from lib.Util import *
import caffe

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class CaffeFeature(object): # for image
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        self.net = caffe.Net(conf['deploy'], conf['model'], caffe.TEST)
#        print self.net
        self.trans = caffe.io.Transformer({'pair_data': self.net.blobs['pair_data'].data.shape})
        self.trans.set_transpose('pair_data', (2,0,1))
        
        if conf.has_key('mean'):
            mean_info = self.loadMean(conf['mean'])
            self.trans.set_mean('pair_data', mean_info.mean(1).mean(1)) # mean pixel
        self.trans.set_raw_scale('pair_data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.trans.set_channel_swap('pair_data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        self.net.blobs['pair_data'].reshape(1,3,227,227)
    
    def levelData(self, level):
        return self.net.blobs[level].data[0]
    
    def process(self, imgfile, level):
        img = loadImg(imgfile)
#        for i in range(3):
#            tmp = img[:, :, i]
#            print tmp.shape
#            skimage.io.imsave("feature_see%d.jpg"%i, tmp)
#        print img
#        print img.shape
#        img[:, :, 2] = 0
        img = self.trans.preprocess('pair_data', img)
#        print img
#        print img.shape
        self.net.blobs['pair_data'].data[...] = img
        if self.conf['gpu']:
            caffe.set_device(0)
            caffe.set_mode_gpu()
        out = self.net.forward()
#        print out
        res = self.levelData(level)
#        print res
#        print len(res)
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.imshow(self.trans.deprocess('data', self.net.blobs['data'].data[0]))
#        plt.show()
#        
#        feat1 = self.net.blobs[level].data[0]
#        self.plot(res, padval=0.5)
        return res

    def loadMean(self, filename):
        filetype = filename.split(".")[-1]
        nparray = False
        if filetype == "npy":
            nparray = np.load(filename)
        else:
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open( filename , 'rb' ).read()
            blob.ParseFromString(data)
            nparray = np.array( caffe.io.blobproto_to_array(blob) ).reshape(3,227,227)
        print nparray.shape
        return nparray
    
    def plot(self, data, padsize=1, padval=0):
        print data.shape
        data -= data.min()
        data /= data.max()

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        skimage.io.imsave("feature_see.jpg", data)
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.imshow(data)
#        plt.show()
#        plt.savefig("feature_see.jpg")
    
    def testPrint(self):
        print "Hello World!"

def main():
    conf = {
        "model": "/datas/root/deepid_test/models/sauron/sauron_iter_320000.caffemodel",
        "mean": "/datas/root/deepid_test/dataset/att_train_mean.binaryproto",
        "deploy": "/datas/root/deepid_test/models/sauron/deploy.prototxt",
        "gpu": False,
    }
    t = CaffeFeature(conf)
    t.process(sys.argv[1], "data")
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

