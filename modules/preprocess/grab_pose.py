#!/usr/bin/env python
"""
This is a helper script that copied over the pose CSV files from Lifeng's folder
"""
print(__doc__)

import os
from shutil import copyfile

source_path = '/media/mo/remoteTitan/home/deeplearning/dataset_on_titan-pascal/topdown_urban/4-27-rotated/meta/'
destination_path = '/home/mo/data_lifeng/sampled_training_data_5-4/poses/'

if not os.path.exists(destination_path):
  os.mkdir(destination_path)

drive_ids = [i for i in os.listdir(source_path) if i[:2]=='HT']
for drive in drive_ids:
  pose_file = drive+'-pose.csv'
  if not os.path.exists(destination_path+pose_file):
    print('Copying %s over to %s'%(source_path+drive+'/'+pose_file,destination_path+pose_file))
    copyfile(source_path+drive+'/'+pose_file,destination_path+pose_file)
  else:
    print('... %s already exists'%(destination_path+pose_file))
  bbox_file = drive+'_bboxlist.txt'
  if not os.path.exists(destination_path+bbox_file):
    print('Copying %s over to %s'%(source_path+drive+'/'+bbox_file,destination_path+bbox_file))
    copyfile(source_path+drive+'/'+bbox_file,destination_path+bbox_file)
  else:
    print('... %s already exists'%(destination_path+bbox_file))

