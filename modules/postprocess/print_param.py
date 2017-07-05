#!/usr/bin/env python
"""
This script prints out parameters for this module
"""
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--config",help="path to config file")
args = parser.parse_args()

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["postprocess"]

if param['stitch_tracklets']:
	print('...Stitching is enabled')
else:
	print('...Stitching is DISABLED')

if param['point_reduction']:
  print('...Polyline point reduction is enabled')
else:
  print('...Polyline point reduction is DISABLED')