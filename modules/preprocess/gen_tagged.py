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

with open(args.config) as fparam:
  parameters = json.loads(jsmin(fparam.read()))["preprocess"]

drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

tag_fmt = ['%d','%.10f','%.10f','%.10f','%d','%02d']
for drive in drive_list:
  print('Working on drive %s'%drive)

  #get and process drive pose
  pose_path = poses_path+drive+'-pose.csv'
  pose = np.loadtxt(pose_path) #format: latitude longitude altitude timestamp
  pose = pose[pose[:,3].argsort(),:] #sort based on timestamp
  scale_meta = motutil.meterize(pose) #warning: pose is modified (meterized) in place
  tmap_pose = {ts_origin:counter*1e6 for counter,ts_origin in enumerate(pose[:,3])}

  #get meta information about image tiles on pose
  bbox_path = poses_path+drive+'_bboxlist.txt'
  bbox_list = np.loadtxt(bbox_path,delimiter=',') #format: minLat, minLon, maxLat, maxLon, chunk, t_start, t_end, lat, lon, alt
  # tend_dict = dict(zip(bbox_list[:,5],bbox_list[:,6])) #mapping from start to end timestamps for each bbox

  #get the list of image annotations on this drive
  filelist = sorted([i for i in os.listdir(input_path) if '_'.join(i.split('_')[:2])==drive])

  for filename in filelist:
    if os.path.exists(output_path+filename) and os.path.exists(output_path+filename+'.tmap')>=parameters['fake_timestamp']:
        # print('\t%s exists! skipping'%(output_path+filename))
        continue
    print('\tworking on %s'%(output_path+filename))

    #crop the pose according to start and end timestamps of bbox (topdown image)
    # timestamp_start = float(filename.split('_')[-1].split('.')[0])
    # timestamp_end = tend_dict[timestamp_start]
    # pose_crop = pose[np.logical_and(pose[:,3]>=timestamp_start-100000000,pose[:,3]<=timestamp_end+100000000),:]
    pose_crop = pose
    timestamp_seed = int(filename.split('_')[-1].split('.')[0])

    points = np.loadtxt(input_path+filename)

    #warning: points are meterized in place
    tagged,tagged_tmap = motutil.get_tagged(points,pose_crop,scale_meta,tmap_pose,parameters)
    np.savetxt(output_path+filename,tagged,fmt=tag_fmt,delimiter=',')
    if parameters['fake_timestamp']:
      np.savetxt(output_path+filename+'.tmap',tagged_tmap,fmt=['%d','%d'],delimiter=',')
