#!/usr/bin/env python
"""
This script assigns timestamps to detections
"""
print(__doc__)

import os
# from PIL import Image
import numpy as np
import argparse
import json
from jsmin import jsmin

import motutil

parser = argparse.ArgumentParser()
parser.add_argument("--input",  help="path to input annotations")
parser.add_argument("--output", help="path to tagged annotations")
parser.add_argument("--config", help="path to config file")
parser.add_argument("--drives", help="path to drives list file")
parser.add_argument("--poses",  help="path to drive pose CSV files")
args = parser.parse_args()
input_path = args.input+'/'
poses_path = args.poses+'/'
output_path = args.output+'/'

if not os.path.exists(output_path):
  os.makedirs(output_path)

# with open(args.config) as fparam:
#   param = json.loads(jsmin(fparam.read()))["preprocess"]

drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

tag_fmt = ['%d','%.10f','%.10f','%d','%02d']

for drive in drive_list:
  print('Working on drive %s'%drive)

  #get the pose file for this drive
  pose_path = poses_path+drive+'-pose.csv'
  pose = np.loadtxt(pose_path)

  #get the list of image annotations on this drive
  filelist = sorted([i for i in os.listdir(input_path) if '_'.join(i.split('_')[:2])==drive])

  for filename in filelist:
    if os.path.exists(output_path+filename):
      print('\t%s exists already! skipping...'%(output_path+filename))
      continue

    points = np.loadtxt(input_path+filename)
    tagged = motutil.tag(points,pose)

    with open(output_path+filename,'w') as fout:
      np.savetxt(fout,tagged,fmt=tag_fmt,delimiter=',')


