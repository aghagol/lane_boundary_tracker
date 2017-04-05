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

import motutil

param = {
  'pixel_size':1., #in meters
  'object_size':10., #in meters
  'image_nrows':100, #minimum size
  'image_ncols':100, #minimum size
}

data_dir = 'JSONs/'
output_dir = './out/'

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
    if motutil.json_to_mot_det(data_dir+drive+'/'+surface, det_out+'det.txt', param):
      print('\tDrive %s_%s too large! Skipping...'%(drive,surface))
      skipped +=1
      os.rmdir(det_out)
      os.rmdir(output_dir+'%s_%s/'%(drive,surface_name))
    else: 
      #save timestamps
      motutil.store_json_timestamps(data_dir+drive+'/'+surface, det_out+'timestamps.txt')
      #save images
      img_out = output_dir+'%s_%s/img1/'%(drive,surface_name)
      os.makedirs(img_out)
      Image.fromarray(np.zeros((param['image_nrows'],param['image_ncols']))).convert('RGB').save(img_out+'000001.jpg')
      #save groundtruth
      gt_out = output_dir+'%s_%s/gt/'%(drive,surface_name)
      os.makedirs(gt_out)
      motutil.json_to_mot_gt(data_dir+drive+'/'+surface, gt_out+'gt.txt', param)
      processed +=1

print('Stats:\n\tSkipped=%d, Processed=%d'%(skipped,processed))


