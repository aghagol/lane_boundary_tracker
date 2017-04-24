#!/usr/bin/env python
"""
This script prints out parameters for this module
"""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config",help="path to config file")
args = parser.parse_args()

with open(args.config) as fparam:
  param = json.load(fparam)["postprocess"]

if param['stitch']:
	print('...Stitching tracklets')
