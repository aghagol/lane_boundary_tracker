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

if param['video_tracking']:
	print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
	print('WARNING: re-sampling timestamps for \"Video Processing\"')
	print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

if param['video_tracking'] and param['fake_dets']:
	print('...Adding up to %d fake points after each detection (about %d meters apart)'%(param['fake_dets_n'],param['pose_step']))
if param['video_tracking'] and param['start_with_pose']:
	print('...Using pose to initialize tracking state parameters')