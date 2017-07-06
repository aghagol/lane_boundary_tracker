#!/usr/bin/env python
"""
This script creates an MOT-like dataset
"""

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
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

if args.verbosity>=2:
  print(__doc__)

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["preprocess"]

drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

for drive in drive_list:
  if args.verbosity>=2:
    print('Working on drive %s'%drive)

  #split the drives on large gaps (cluster the images)
  filelist = sorted([i for i in os.listdir(args.input) if ('_'.join(i.split('_')[:2])==drive and i.endswith('.fuse'))])
  if param['split_on_temporal_gaps']:
    if args.verbosity>=2:
      print('\tClustering images based on time overlap...')
    drive_parts = []
    for filename in filelist:
      if os.stat(args.input+'/'+filename).st_size:
        file_ismember = False
        points = np.loadtxt(args.input+'/'+filename,delimiter=',').reshape(-1,4)
        t0 = points[:,0].min()*1e-6
        t1 = points[:,0].max()*1e-6
        for part in drive_parts:
          if (part['t0']-t0<param['gap_min'] and t0-part['t1']<param['gap_min']) or \
             (part['t0']-t1<param['gap_min'] and t1-part['t1']<param['gap_min']):
            part['members'].append(filename)
            part['t0'] = min(part['t0'],t0)
            part['t1'] = max(part['t1'],t1)
            part['n'] += points.shape[0]
            file_ismember = True
        if not file_ismember:
          drive_parts.append({'t0':t0,'t1':t1,'members':[filename],'n':points.shape[0]})
    clusters = {'%s_part_%05d'%(drive,i):part['members'] for i,part in enumerate(drive_parts) if part['n']>=param['min_seq_size']}

    #log
    for i,part in enumerate(drive_parts):
      if part['n']<param['min_seq_size']:
        if args.verbosity>=2:
          print('\t...discarding %s_part_%05d due to small size (%d<%d)'%(drive,i,part['n'],param['min_seq_size']))
      else:
        if args.verbosity>=2:
          print('\t...subdrive %s_part_%05d has the following members:'%(drive,i))
        for member in sorted(part['members']):
          if args.verbosity>=2:
            print('\t\t%s'%(member))
  else:
    clusters = {drive:filelist}

  #keep track of subdrives with 0 or 1 detection points that will be marked for deletion
  #these subdrives go unseen until now because points can be removed in later stages
  tiny_subdrives = set()

  motutil.index_TLLA_points(args.input,args.output,clusters,tiny_subdrives,param)
  motutil.ss_to_mot_det(args.output,clusters,tiny_subdrives,args.poses+'/'+drive+'-pose.csv',param)
  
  #save groundtruth
  if param['generate_gt']:
    motutil.ss_to_mot_gt(args.output,clusters,tiny_subdrives,args.poses+'/'+drive+'-pose.csv',param)

  # #save (fake) images
  # img_out = args.output+'/%s_%s/img1/'%(drive,surface_name)
  # os.makedirs(img_out)
  # Image.fromarray(np.zeros((param['image_nrows'],param['image_ncols']))).convert('RGB').save(img_out+'000001.jpg')
