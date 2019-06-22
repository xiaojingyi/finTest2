#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: cache.py
# Date: 2016 Wed 05 Oct 2016 02:55:03 PM CST
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: 
#   This just works with redis

__author__="Jingyi"

import os, sys, time
import hashlib
import redis
import json

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

class Cache(object):
    def __init__(self, config):
        if not config:
            self.bail(-1, "no config: cache init")
        self.config = config
        self.debug = config["debug"]
        if not config.has_key("host"):
            self.config["host"] = "localhost"
        if not config.has_key("prefix"):
            self.prefix = "c_"
        else:
            self.prefix = config["prefix"]
        self.c = redis.Redis(host=self.config["host"], port=6379)
        #super(cache, self).__init__(config)
    
    def jsonSet(self, key, value, expr=0):
        value = json.dumps(value)
        self.set(key, value, expr, False)
        return 

    def jsonGet(self, key, use_hash=False):
        res = self.get(key, False)
        if res:
            res = json.loads(res)
        return res

    def set(self, key, value, expr=0, use_hash=False):
        if use_hash:
            key = self._key(key, self.prefix)
        self.c.set(self.prefix + key, value, ex=expr)
        return

    def get(self, key, use_hash=False):
        if use_hash:
            key = self._key(key, self.prefix)
        return self.c.get(self.prefix + key)

    def _key(self, name, prefix="ts_"):
        m = hashlib.md5()
        m.update(name)
        return prefix + str(m.hexdigest())

    def testPrint(self):
        print "Hello World!"

    def bail(self, sig, msg):
        print sig, ": ", msg
        exit()

def main():
    conf = {
            "debug": True,
            "host": "localhost",
            "prefix": "t_",
            }
    t = Cache(conf)
    t.jsonSet("test", {"love": 0}, expr=3)
    print t.jsonGet("test")
    time.sleep(5)
    print t.jsonGet("test")
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

