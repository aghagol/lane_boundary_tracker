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

with open(args.config) as fparam:
  param = json.load(fparam)["preprocess"]
print(param)

#if "drive_list.txt" is not found in data root folder, then generate one (from all drives)
prefix = '/Lane/sampled_fuse/'
image_list = [i[:-8] for i in os.listdir(data_dir+prefix) if i.endswith('.png.txt')]
if not os.path.exists(data_dir+'/drive_list.txt'):
  drive_list = sorted(set(['_'.join(imagename.split('_')[:2]) for imagename in image_list]))
  with open(data_dir+'/drive_list.txt','w') as fdrivelist:
    for drive in drive_list:
      fdrivelist.write('%s\n'%(drive))
else:
  drive_list = []
  with open(data_dir+'/drive_list.txt') as fdrivelist:
    for line in fdrivelist:
      drive_list.append(line.strip())

for drive in drive_list:
  print('Working on drive %s'%drive)
  pose_path = data_dir+'/poses/'+drive+'.csv'

  det_out = output_dir+'%s/det/'%(drive)
  os.makedirs(det_out)

  motutil.index_TLLA_points(data_dir,det_out+'tlla.txt',drive,param)
  motutil.ss_to_mot_det(det_out+'tlla.txt',pose_path,det_out+'det.txt',param)
  
  # #save groundtruth
  # gt_out = output_dir+'%s_%s/gt/'%(drive,surface_name)
  # os.makedirs(gt_out)
  # motutil.highv1_to_mot_gt(data_dir+drive+'/'+surface, gt_out+'gt.txt', param)

  # #save (fake) images
  # img_out = output_dir+'%s_%s/img1/'%(drive,surface_name)
  # os.makedirs(img_out)
  # Image.fromarray(np.zeros((param['image_nrows'],param['image_ncols']))).convert('RGB').save(img_out+'000001.jpg')

