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
parser.add_argument("--input",help="path to input drive")
parser.add_argument("--output",help="output path to MOT dataset")
parser.add_argument("--config",help="path to config file")
parser.add_argument("--drives",help="path to drives list file")
args = parser.parse_args()
data_dir = args.input+'/'
output_dir = args.output+'/'

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["preprocess"]
print(param)
print("")

#if "drive_list.txt" is not found in data root folder, then generate one (from all drives)
if os.path.exists(args.drives):
  drive_list = []
  with open(args.drives) as fdrivelist:
    for line in fdrivelist:
      drive_list.append(line.strip())
else:
  print("Generating a drive list because it is missing!")
  prefix = '/Lane/sampled_fuse/'
  image_list = [i[:-8] for i in os.listdir(data_dir+prefix) if i.endswith('.png.txt')]
  drive_list = sorted(set(['_'.join(imagename.split('_')[:2]) for imagename in image_list]))
  with open(args.drives,'w') as fdrivelist:
    for drive in drive_list:
      fdrivelist.write('%s\n'%(drive))

for drive in drive_list:
  print('Working on drive %s'%drive)

  #read pose information
  pose_path = data_dir+'/poses/'+drive+'-pose.csv'

  #split the drives on large gaps
  prefix = '/Lane/sampled_fuse/'
  filelist = sorted([i for i in os.listdir(data_dir+prefix) if '_'.join(i.split('_')[:2])==drive])
  if param['split_on_temporal_gaps']:
    drive_parts = []
    for filename in filelist:
      if os.stat(data_dir+prefix+filename).st_size:
        file_ismember = False
        points = np.loadtxt(data_dir+prefix+filename,delimiter=',').reshape(-1,5)
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
    os.makedirs(output_dir+'%s/det/'%(subdrive))

  motutil.index_TLLA_points(data_dir,output_dir,clusters,param)
  motutil.ss_to_mot_det(output_dir,clusters,pose_path,param)
  
  # #save groundtruth
  # gt_out = output_dir+'%s_%s/gt/'%(drive,surface_name)
  # os.makedirs(gt_out)
  # motutil.highv1_to_mot_gt(data_dir+drive+'/'+surface, gt_out+'gt.txt', param)

  # #save (fake) images
  # img_out = output_dir+'%s_%s/img1/'%(drive,surface_name)
  # os.makedirs(img_out)
  # Image.fromarray(np.zeros((param['image_nrows'],param['image_ncols']))).convert('RGB').save(img_out+'000001.jpg')
