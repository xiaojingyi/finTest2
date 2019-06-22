#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2013 Jingyi Xiao
#
# Encoding: UTF-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Created time: $2015-9-15 15:22:38$
# Note: This source file is NOT a freeware
# Version: buildnet.py 0.1 jingyi Exp $

__author__="jingyi"
__date__ ="$2015-9-15 15:22:38$"

import os, sys, time
os.environ['NLS_LANG'] = "SIMPLIFIED CHINESE_CHINA.UTF8"
sys.path.append("../")
sys.path.append("/datas/lib/py")
from lib.Util import *
from model.NetConstructor import NetConstructor
import numpy as np

def main():
    config = {
        "debug": True,
    }
    data_size = 51529
    split_num = 227
    one_num = 168
    net = [
        {
            "type": "data",
            "data_type": "sauron_data",
        },
        {
            "name": "data-split1",
            "source": "flat_data",
            "source_p": "flat_data_p",
            "type": "split",
            "size_all": data_size,
            "dim": 1,
            
            "number": split_num,
            "layers": [200, one_num],
            "neurons": ["relu", "relu", ],
            "drops": [0, 0, 0, 0],
            "helper_loss": 0,
            "concats": {
                "all": True,
            },
        },
        {
            "name": "data-split2",
            "source": "data-split1",
            "source_p": "data-split1_p",
            "type": "split",
            "size_all": 227*168,
            "dim": 1,
            "number": 168,
            "layers": [168],
            "neurons": ["relu", False, ],
            "drops": [0, 0, 0, 0],
            "helper_loss": 0,
            "concats": {
                "all": True,
            },
        },
        {
            "name": "data-split3",
            "source": "data-split2",
            "source_p": "data-split2_p",
            "type": "split",
            "size_all": 168*168,
            "dim": 1,
            "number": 128,
            "layers": [168],
            "neurons": ["relu", False, ],
            "drops": [0, 0, 0, 0],
            "helper_loss": 0,
            "concats": {
                "all": True,
            },
        },
        {
            "name": "data-split4",
            "source": "data-split3",
            "source_p": "data-split3_p",
            "type": "split",
            "size_all": 168*128,
            "dim": 1,
            "number": 128,
            "layers": [128],
            "neurons": ["relu", False, ],
            "drops": [0, 0, 0, 0],
            "helper_loss": 0,
            "concats": {
                "all": True,
            },
        },
        {
            "name": "data-split5",
            "source": "data-split4",
            "source_p": "data-split4_p",
            "type": "split",
            "size_all": 128*128,
            "dim": 1,
            "number": 80,
            "layers": [128],
            "neurons": ["relu", False, ],
            "drops": [0, 0, 0, 0],
            "helper_loss": 0,
            "concats": {
                "all": True,
            },
        },
        {
            "name": "data-split6",
            "source": "data-split5",
            "source_p": "data-split5_p",
            "type": "split",
            "size_all": 80*128,
            "dim": 1,
            "number": 80,
            "layers": [80],
            "neurons": ["relu", False, ],
            "drops": [0, 0, 0, 0],
            "helper_loss": 0,
            "concats": {
                "all": True,
            },
        },
        {
            "name": "data-split7",
            "source": "data-split6",
            "source_p": "data-split6_p",
            "type": "split",
            "size_all": 80*80,
            "dim": 1,
            "number": 60,
            "layers": [80],
            "neurons": ["relu", False, ],
            "drops": [0, 0, 0, 0],
            "helper_loss": 0,
            "concats": {
                "all": True,
            },
        },
        {
            "source": "data-split7",
            "source_p": "data-split7_p",
            "type": "ips",
            "neurons": ["relu", "relu", "relu", ],
            "drops": [0, 0.3, 0.3],
            "layers": [2048, 2048, 2048, ],
            "stds": [],
            "preffix": "fc",
        },
        {
            "type": "loss",
            "loss_type": "simloss",
            "params": {
                "name": "sim-loss",
                "id1": "fc-2",
                "id2": "fc-2_p",
                "sim": "sim",
                "loss_weight": "0.3",
            },
        },
        {
            "source": "fc-2",
            "type": "ip",
            "relu": False,
            "drop": 0,
            "num": 8000,
            "is_classify": True,
            "name": "out",
            "classify_loss": 1,
            "params": False,
        },
        {
            "source": "fc-2_p",
            "type": "ip",
            "relu": False,
            "drop": 0,
            "num": 8000,
            "is_classify": True,
            "name": "out_p",
            "classify_loss": 1,
            "params": ["out_w", "out_b"],
        },
        {
#            "source": "fc-6-sigmoid",
            "source": "fc-0",
            "diff": "flat_data",
            "type": "help",
            "size_all": data_size,
            "number": split_num,
            "one_num": one_num,
            "layers": [2048, 2048, ],
            "neurons": ["sigmoid", "sigmoid", False, False, ],
            "drops": [0, 0, 0, 0, ],
            "preffix": "help",
            "loss_weight": 0.01,
        },
    ]
    model = NetConstructor(config)
    model.netDefine(net)
    model.writeNet("/datas/codes/sauron/Trainer/run/sauron_tree.prototxt")
    return

if __name__ == "__main__":
    main()
