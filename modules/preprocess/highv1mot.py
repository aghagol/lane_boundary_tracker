#!/usr/bin/env python
"""
This script creates an MOT-like dataset
Input format: Chen's format
MOT format: a CSV with columns: frame#, -1, x, y, w, h, confidence, -1, -1, -1
"""
print(__doc__)

import os
from PIL import Image
import numpy as np
import argparse
import json

import motutil

parser = argparse.ArgumentParser()
parser.add_argument("--input",help="path to input drive")
parser.add_argument("--output",help="output path to MOT dataset")
parser.add_argument("--config",help="path to config file")
args = parser.parse_args()
data_dir = args.input+'/'
output_dir = args.output+'/'

'''
param:
  pixel_size.........................in meters
  object_size........................in meters
  fake_dets..........................False/True
'''
with open(args.config) as fparam:
  param = json.load(fparam)["preprocess"]
print(param)

skipped = 0
processed = 0
for drive in os.listdir(data_dir):
  print('Working on drive %s'%drive)

  pose_filename = [csv_file for csv_file in os.listdir(data_dir+drive) if csv_file.endswith('csv')]
  if not len(pose_filename)==1: exit('Exactly one pose file is allowed!')
  pose_filename = pose_filename[0]

  det_out = output_dir+'%s/det/'%(drive)
  os.makedirs(det_out)

  state = motutil.highv1_to_mot_det(data_dir+drive, pose_filename, det_out+'det.txt', param)

  #save timestamps
  motutil.store_highv1_timestamps(data_dir+drive, pose_filename, det_out+'timestamps.txt')

  # #save groundtruth
  # gt_out = output_dir+'%s_%s/gt/'%(drive,surface_name)
  # os.makedirs(gt_out)
  # motutil.json_to_mot_gt(data_dir+drive+'/'+surface, gt_out+'gt.txt', param)

  # #save images
  # img_out = output_dir+'%s_%s/img1/'%(drive,surface_name)
  # os.makedirs(img_out)
  # Image.fromarray(np.zeros((param['image_nrows'],param['image_ncols']))).convert('RGB').save(img_out+'000001.jpg')

