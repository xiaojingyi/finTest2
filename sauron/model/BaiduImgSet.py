#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: BaiduImgSet.py
# Date: 2015 2015年05月22日 星期五 22时06分45秒
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
import json
from ImgSet import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class BaiduImgSet(ImgSet):
    def __init__(self, conf):
        self.conf = {}
        self.conf = conf
        super(BaiduImgSet, self).__init__(conf)
    
    def loadsData(self, data):
        res = False
        try:
            res = json.loads(data)
            self.last_res = res
        except:
            res = self.last_res
            print "data err"
            exit()
        return res

    def pickImg(self, data, t, i):
        data_len = len(data[self.data_key])
        if data_len < 10:
            print data
            return False
        else:
            print data_len
        pic = {}
        pic["cat"] = self.s["col"]
        pic["spec"] = self.s["tag"]
        pic["type"] = t
        for one in data[self.data_key]:
            if one:
                pic["info"] = one["desc"]
                try:
                    pic["uri"] = one["imageUrl"]
                    pic["width"] = one["imageWidth"]
                    pic["height"] = one["imageHeight"]
                except:
                    pic["uri"] = one["image_url"]
                    pic["width"] = one["image_width"]
                    pic["height"] = one["image_height"]

                pic["md5"] = md5(pic["uri"])
                pic["tags"] = ",".join(one["tags"])
                #print printJson(pic)
                self.save(pic, i)
        return True

    def testPrint(self):
        return

    def test(self, uri, referer, data_key, datas):
        self.uriTemplate(uri, referer, data_key)
        self.setUriParam(datas)
        res = self.fetchSet(0, 20)
        print len(res[self.data_key])
        return

    def run(self, uri, referer, data_key, datas, t, start=0, to=100):
        self.uriTemplate(uri, referer, data_key)
        self.setUriParam(datas)
        for i in range(start, to):
            res = self.fetchSet(i * 60, 60)
            ret = self.pickImg(res, t, i)
            if not ret:
                break
            time.sleep(1)
        
        return

    def getMeinv(self):
        uri = "http://image.baidu.com/data/imgs?col=<col>&tag=<tag>&sort=0&tag3=&pn=<start>&rn=<limit>&p=channel&from=1"
        referer = "http://image.baidu.com/channel?c=<col>"
        data_key = "imgs"
        datas = {
                "col": "美女",
                "tag": "全部",
                "tag3": "",
        }
        #self.test(uri, referer, data_key, datas)
        self.run(uri, referer, data_key, datas, "meinv", 0, 1000)
        return

    def getStar(self):
        uri = "http://image.baidu.com/data/star/listjson?pn=<start>&rn=<limit>&name=<tag>&sorttype=0&p=star.home&col=<col>&tag=<tag3>"
        referer = "http://image.baidu.com/channel/star/<col>"
        data_key = "data"
        from StarName import StarName
        sn = StarName({})
        male = sn.getNamesMale()
        female = sn.getNamesFemale()
        for one in male:
            datas = { "col": "明星", "tag": one, "tag3": one, }
            self.run(uri, referer, data_key, datas, "star_male", 0, 100)
        for one in female:
            datas = { "col": "明星", "tag": one, "tag3": one, }
            self.run(uri, referer, data_key, datas, "star_female", 0, 100)
        return

    def getStarRandom(self):
        uri = "http://image.baidu.com/data/star/listjson?pn=<start>&rn=<limit>&name=<tag>&sorttype=0&p=star.cate&col=<col>&tag=<tag3>"
        referer = "http://image.baidu.com/channel/star/cate/<col>"
        data_key = "data"
        datas = {
                "col": "明星",
                "tag": "明星写真",
                "tag3": "明星写真",
        }
        #self.test(uri, referer, data_key, datas)
        self.run(uri, referer, data_key, datas, "stars", 0, 200)
        return

    def getJiaju(self):
        # city & house
        uri = "http://image.baidu.com/data/imgs?col=<col>&tag=<tag>&sort=20&tag3=<tag3>&pn=<start>&rn=<limit>&p=channel&from=1"
        referer = "http://image.baidu.com/channel?c=<col>&t=<tag>&s=20&t3=<tag3>"
        data_key = "imgs"
        datas = {
                "col": "家居",
                "tag": "热门推荐",
                "tag3": "别墅",
        }
        #self.test(uri, referer, data_key, datas)
        self.run(uri, referer, data_key, datas, "jiajuzhuang", 0, 1000)

def main():
    conf = {
            "db_host": "localhost",
            "db_user": "root",
            "db_pass": "",
            "db_database": "baiduimg",
    }
    t = BaiduImgSet(conf)
    #t.getStar()
    #t.getStarRandom()
    t.getMeinv()
    t.getJiaju()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

