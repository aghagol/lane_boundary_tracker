#!/usr/bin/env python
"""
This script prints out parameters to log
"""
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--config",help="path to config file")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["preprocess"]

#log
if args.verbose: print("\nParamteres:")
for param_key,param_val in param.iteritems():
  if args.verbose: print('...%s = %s'%(param_key,param_val))
