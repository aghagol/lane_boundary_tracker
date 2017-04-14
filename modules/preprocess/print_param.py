#!/usr/bin/env python
"""
This script prints out parameters that were used for JSON to MOT conversion
"""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--param",help="path to parameters JSON file")
args = parser.parse_args()

with open(args.param) as fparam:
  param = json.load(fparam)

print('Config: Sampling interval = %d meters'%(param['step_size']))
print('Config: Recall rate = %04.2f%%'%(param['recall']*100))