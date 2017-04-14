#!/usr/bin/env python
"""
This script creates an MOT-like dataset
Input format: a JSON file
MOT format: a CSV with columns: frame#, -1, x, y, w, h, confidence, -1, -1, -1
"""
print(__doc__)

import os
from PIL import Image
import numpy as np
import argparse

import motutil

parser = argparse.ArgumentParser()
parser.add_argument("--input",help="path to input JSONs")
parser.add_argument("--output",help="output path to MOT dataset")
args = parser.parse_args()
data_dir = args.input+'/'
output_dir = args.output+'/'

param = {
  'step_size':3, #pose subsampling
  'pixel_size':1., #in meters
  'object_size':10., #in meters
  'image_nrows':100, #minimum size
  'image_ncols':100, #minimum size
  'min_dets':10, #minimum #dets on a sequence to be included
  'recall':.7, #recall ratio
  'keep':3, #at the start of seq, keep this many perfect frames
}

print(param)

skipped = 0
processed = 0
for drive in os.listdir(data_dir):
  print('Working on drive %s'%drive)
  for surface in os.listdir(data_dir+drive):
    if not surface.endswith('json'):
      continue
    else:
      surface_name = surface[:-5]
    det_out = output_dir+'%s_%s/det/'%(drive,surface_name)
    os.makedirs(det_out)
    state = motutil.json_to_mot_det(data_dir+drive+'/'+surface, det_out+'det.txt', param)
    if state==1:
      print('\tDrive %s_%s: too large! Skipping image generation.'%(drive,surface))
      #save timestamps
      motutil.store_json_timestamps(data_dir+drive+'/'+surface, det_out+'timestamps.txt')
      #save groundtruth
      gt_out = output_dir+'%s_%s/gt/'%(drive,surface_name)
      os.makedirs(gt_out)
      motutil.json_to_mot_gt(data_dir+drive+'/'+surface, gt_out+'gt.txt', param)
    elif state==2:
      print('\tDrive %s_%s: too few detections! Skipping the sequence.'%(drive,surface))
      skipped +=1
      os.rmdir(det_out)
      os.rmdir(output_dir+'%s_%s/'%(drive,surface_name))
    elif state==0:
      #save timestamps
      motutil.store_json_timestamps(data_dir+drive+'/'+surface, det_out+'timestamps.txt')
      #save groundtruth
      gt_out = output_dir+'%s_%s/gt/'%(drive,surface_name)
      os.makedirs(gt_out)
      motutil.json_to_mot_gt(data_dir+drive+'/'+surface, gt_out+'gt.txt', param)
      #save images
      img_out = output_dir+'%s_%s/img1/'%(drive,surface_name)
      os.makedirs(img_out)
      Image.fromarray(np.zeros((param['image_nrows'],param['image_ncols']))).convert('RGB').save(img_out+'000001.jpg')

