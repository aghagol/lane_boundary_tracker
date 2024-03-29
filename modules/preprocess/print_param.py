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
  param = json.load(fparam)["preprocess"]

print('...Sampling interval = %d meters'%(param['step_size']))
print('...Recall rate = %04.2f%%'%(param['recall']*100))