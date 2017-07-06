#!/usr/bin/env python
"""
This script prints out parameters for this module
"""

import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--config",     help="path to config file")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["postprocess"]

if args.verbosity>=1:
  if param['stitch_tracklets']:
  	print('...Stitching is enabled')
  else:
  	print('...Stitching is DISABLED')
  if param['point_reduction']:
    print('...Polyline point reduction is enabled')
  else:
    print('...Polyline point reduction is DISABLED')

if args.verbosity>=2:
  print("\nParamteres:")
  for param_key,param_val in param.iteritems():
    print('...%s = %s'%(param_key,param_val))
