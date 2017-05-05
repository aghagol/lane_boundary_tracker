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
  param = json.loads(jsmin(fparam.read()))["visualize"]

print('...Omitting tracks shorter than %d'%(param['min_track_length']))
