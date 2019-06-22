#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: DayMakerTrans2Mongo.py
# Date: 2016 Tue 04 Oct 2016 04:44:04 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
sys.path.append("/datas/lib/py")
sys.path.append("/.jingyi/codes/trader")
from classes.DMaker import DMaker
from libs.data.DInsertor import DInsertor

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class DayMakerTrans2Mongo(DMaker):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: DayMakerTrans2Mongo init")
        self.config = config
        self.debug = config["debug"]
        super(DayMakerTrans2Mongo, self).__init__(config)
    
    def run(self):
        data, index_sh = self.rangeDaysAll(
                self.config['dates'][0],
                self.config['dates'][1],
                )
        rg = self.time_range
        print len(data)
        print len(data[0])
        print len(data[0][0])
        print len(index_sh)
        print len(rg)
        print "Hello World!"

def main():
    conf = {
            "debug": True,
            "thread_num": 26,

            # current pro data
            "dates": ['2008-01-01', '2016-05-28'],
            "trdates": ['2013-01-02', '2016-05-28'],
            "tdates": ['2010-01-01', '2013-01-01'],

            }
    t = DayMakerTrans2Mongo(conf)
    t.run()
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

