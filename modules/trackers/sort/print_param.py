#!/usr/bin/env python
"""
This script prints out parameters that were used for JSON to MOT conversion
"""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config",help="path to config file")
args = parser.parse_args()

with open(args.config) as fparam:
  param = json.load(fparam)["sort"]

print('...IOU threshold - high = %f'%(param['iou_threshold_high']))
print('...IOU threshold - low = %f'%(param['iou_threshold_low']))
print('...Max age = %d'%(param['max_age']))
print('...Min hits = %d'%(param['min_hits']))
