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
  param = json.loads(jsmin(fparam.read()))["sort"]

print('...Detections to tracks matching distance threshold - tight = %.2f meters'%(param['d2t_dist_threshold_tight']))
print('...Detections to tracks matching distance threshold - loose = %.2f meters'%(param['d2t_dist_threshold_loose']))
print('...Max age (after last update) = %d frames'%(param['max_age_after_last_update']))
