#!/usr/bin/env python
"""
This script creates an MOT-like dataset
"""
print(__doc__)

import os
from PIL import Image
import numpy as np
import argparse
import json
from jsmin import jsmin

import motutil

parser = argparse.ArgumentParser()
parser.add_argument("--input",  help="path to tagged annotations")
parser.add_argument("--output", help="output path to MOT dataset")
parser.add_argument("--config", help="path to config file")
parser.add_argument("--drives", help="path to drives list file")
parser.add_argument("--poses",  help="path to drive pose CSV files")
args = parser.parse_args()
input_path = args.input+'/'
poses_path = args.poses+'/'
output_path = args.output+'/'

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["preprocess"]
print(param)
print("")

drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

for drive in drive_list:
  print('Working on drive %s'%drive)

  #split the drives on large gaps (cluster the images)
  filelist = sorted([i for i in os.listdir(input_path) if '_'.join(i.split('_')[:2])==drive])
  if param['split_on_temporal_gaps']:
    drive_parts = []
    for filename in filelist:
      if os.stat(input_path+filename).st_size:
        file_ismember = False
        points = np.loadtxt(input_path+filename,delimiter=',').reshape(-1,5)
        t0,t1 = points[:,0].min(),points[:,0].max()
        for part in drive_parts:
          if (part['t0']-t0<param['gap_min'] and t0-part['t0']<param['gap_min']) or \
             (part['t1']-t1<param['gap_min'] and t1-part['t1']<param['gap_min']):
            part['members'].append(filename)
            part['t0'] = min(part['t0'],t0)
            part['t1'] = max(part['t1'],t1)
            part['n'] += points.shape[0]
            file_ismember = True
        if not file_ismember:
          drive_parts.append({'t0':t0,'t1':t1,'members':[filename],'n':points.shape[0]})
    clusters = {'%s_part_%05d'%(drive,i):part['members'] for i,part in enumerate(drive_parts) if part['n']>=param['min_seq_size']}
  else:
    clusters = {drive:filelist}

  for subdrive in clusters:
    os.makedirs(output_path+'%s/det/'%(subdrive))

  motutil.index_TLLA_points(input_path,output_path,clusters,param)
  motutil.ss_to_mot_det(output_path,clusters,poses_path+drive+'-pose.csv',param)
  
  # #save groundtruth
  # gt_out = output_path+'%s_%s/gt/'%(drive,surface_name)
  # os.makedirs(gt_out)
  # motutil.highv1_to_mot_gt(input_path+drive+'/'+surface, gt_out+'gt.txt', param)

  # #save (fake) images
  # img_out = output_path+'%s_%s/img1/'%(drive,surface_name)
  # os.makedirs(img_out)
  # Image.fromarray(np.zeros((param['image_nrows'],param['image_ncols']))).convert('RGB').save(img_out+'000001.jpg')