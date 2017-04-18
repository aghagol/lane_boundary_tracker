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

if param['fake_dets']:
	print('...Adding %d fake points after each detection (about %d meters apart)'%(param['fake_dets_n'],param['pose_step']))
if param['with_pose']:
	print('...Including pose points as detections')