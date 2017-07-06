#!/usr/bin/env python
"""
This script prints out parameters that were used for JSON to MOT conversion
"""
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--config",     help="path to config file")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["visualize"]

if args.verbosity>=1:
  print('...Tracks shorter than %d will be omitted'%(param['min_track_length']))

if args.verbosity>=2:
  print("\nParamteres:")
  for param_key,param_val in param.iteritems():
    print('...%s = %s'%(param_key,param_val))

