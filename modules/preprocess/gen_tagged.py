#!/usr/bin/env python
"""
This script assigns timestamps to detections
"""
print(__doc__)

import os
# from PIL import Image
import numpy as np
import pandas as pd
import argparse
import json
from jsmin import jsmin

import motutil

parser = argparse.ArgumentParser()
parser.add_argument("--input",  help="path to fuse files")
parser.add_argument("--output", help="path to tagged annotations")
parser.add_argument("--images", help="path to image metadata")
parser.add_argument("--config", help="path to config file")
parser.add_argument("--drives", help="path to drives list file")
parser.add_argument("--poses",  help="path to drive pose CSV files")
args = parser.parse_args()
input_path = args.input+'/'
output_path = args.output+'/'
poses_path = args.poses+'/'

if not os.path.exists(output_path):
  os.makedirs(output_path)

with open(args.config) as fparam:
  parameters = json.loads(jsmin(fparam.read()))["preprocess"]

drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

tag_fmt = ['%d','%.10f','%.10f','%.10f']
for drive in drive_list:
  print('Working on drive %s'%drive)

  #get and process drive pose
  pose_path = poses_path+drive+'-pose.csv'
  pose = np.loadtxt(pose_path) #format: latitude longitude altitude timestamp
  pose = pose[pose[:,3].argsort(),:] #sort based on timestamp
  pose_meterized, scale_meta = motutil.meterize(pose)
  pose_tmap = {ts_origin:counter*1e6 for counter,ts_origin in enumerate(pose[:,3])}

  #get meta information about topdown images
  meta = pd.read_csv(os.path.join(args.images,drive,'meta.csv'), skipinitialspace=True) #format: name, time_start, time_end, min_lat, min_lon, max_lat, max_lon

  #get the list of image annotations on this drive
  filelist = sorted([i for i in os.listdir(input_path) if '_'.join(i.split('_')[:2])==drive])

  for filename in filelist:
    if os.path.exists(output_path+filename) and os.path.exists(output_path+filename+'.tmap')>=parameters['fake_timestamp']:
      # print('\t%s exists! skipping'%(output_path+filename))
      continue
    print('\tworking on %s'%(output_path+filename))
    points = np.loadtxt(input_path+filename)

    #clip the pose according to the topdown image
    if parameters['pose_filter_bbox']:
      image_tag = int(filename.split('_')[-1].split('.')[0])
      image_idx = list(meta['time_start']).index(image_tag)
      min_lat = meta['min_lat'][image_idx]
      min_lon = meta['min_lon'][image_idx]
      max_lat = meta['max_lat'][image_idx]
      max_lon = meta['max_lon'][image_idx]
      seed_idx = np.argmax(pose[:,3]>(meta['time_start'][image_idx]+meta['time_end'][image_idx])/2)
      edge_idx = []
      for step in [-1,1]:
        i = seed_idx
        while pose[i,0]>min_lat and pose[i,0]<max_lat and pose[i,1]>min_lon and pose[i,1]<max_lon:
          i +=step
        edge_idx.append(i-step)
      pose_filtered = pose_meterized[edge_idx[0]:edge_idx[1],:]
    else:
      pose_filtered = pose_meterized

    #warning: points are meterized in place
    tagged,tagged_tmap = motutil.get_tagged(points,pose_filtered,pose_tmap,scale_meta,parameters)
    np.savetxt(output_path+filename,tagged,fmt=tag_fmt,delimiter=',')
    if parameters['fake_timestamp']:
      np.savetxt(output_path+filename+'.tmap',tagged_tmap,fmt=['%d','%d'],delimiter=',')
