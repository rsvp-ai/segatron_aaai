#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time     :   2020-04-29 02:43
# @Author   :   Richard Bai
# @EMail    :   he.bai@uwaterloo.ca 
import GPUtil
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index", default='16', type=int, required=True)
args = parser.parse_args()
deviceIDs = []
while deviceIDs==[]:
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.1, maxMemory = 0.1, includeNan=False, excludeID=[], excludeUUID=[])
    if deviceIDs != []:
        index = args.index%len(deviceIDs)
        avaliable_device = deviceIDs[index]
        if len(deviceIDs)>=3:
            time.sleep(60)
        else:
            time.sleep(300)
        deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.1, maxMemory=0.1, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        if deviceIDs!=[] and deviceIDs[index]==avaliable_device:
            break
        else:
            deviceIDs = []
    else:
        time.sleep(60)

print(deviceIDs[0])