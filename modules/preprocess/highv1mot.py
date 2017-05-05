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
import pandas as pd
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

with open(args.config) as fparam:
  param = json.load(fparam)["preprocess"]
print(param)

for drive in os.listdir(data_dir):
  print('Working on drive %s'%drive)

  pose_path = [csv_file for csv_file in os.listdir(data_dir+drive) if csv_file.endswith('pose.csv')]
  assert len(pose_path)==1, 'ERROR: found %d pose files in %s'%(len(pose_path),data_dir+drive)
  pose_path = data_dir+drive+'/'+pose_path[0]

  chunk_times_file = [csv_file for csv_file in os.listdir(data_dir+drive) if csv_file.endswith('_image.csv')]
  assert len(chunk_times_file)==1, 'ERROR: found %d _image.csv files in %s'%(len(chunk_times_file),data_dir+drive)
  chunk_times_file = data_dir+drive+'/'+chunk_times_file[0]
  chunk_times = pd.read_csv(chunk_times_file)
  chunk_times.rename(columns=lambda x: x.strip(),inplace=True) #remove whitespace from headers

  det_out = output_dir+'%s/det/'%(drive)
  os.makedirs(det_out)

  motutil.index_TLLA_points(data_dir+drive,det_out+'tlla.txt',chunk_times,param)
  motutil.highv1_to_mot_det(det_out+'tlla.txt',pose_path,det_out+'det.txt',param)

  #save timestamps
  # motutil.store_highv1_timestamps(pose_path,det_out+'timestamps.txt',param)

  # #save groundtruth
  # gt_out = output_dir+'%s_%s/gt/'%(drive,surface_name)
  # os.makedirs(gt_out)
  # motutil.highv1_to_mot_gt(data_dir+drive+'/'+surface, gt_out+'gt.txt', param)

  # #save (fake) images
  # img_out = output_dir+'%s_%s/img1/'%(drive,surface_name)
  # os.makedirs(img_out)
  # Image.fromarray(np.zeros((param['image_nrows'],param['image_ncols']))).convert('RGB').save(img_out+'000001.jpg')
