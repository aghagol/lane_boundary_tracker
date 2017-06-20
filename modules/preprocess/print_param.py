#!/usr/bin/env python
"""
This script prints out parameters that were used for JSON to MOT conversion
"""
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--config",help="path to config file")
args = parser.parse_args()

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["preprocess"]

print('...Detections will be converted to MOT format')

if param['remove_adjacent_points']:
  print('...Detection points closer than %.2f meters will be removed'%(param['min_pairwise_dist']))

if param['recall']<1:
  print('...Recall= %.2f %%'%(param['recall']*100))