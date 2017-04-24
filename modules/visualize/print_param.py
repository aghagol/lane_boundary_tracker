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
  param = json.load(fparam)["visualize"]

print('...Omitting tracks that span less than %d frames'%(param['min_track_length']))

if param['output_fuse']:
	print('...Converting tracking results to fuse format')
