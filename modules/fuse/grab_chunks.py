#!/usr/bin/env python
"""
This is a helper script that copied over the pose CSV files from Lifeng's folder
"""
print(__doc__)

import os
from shutil import copyfile

source_path = '/media/mo/remoteTitan/home/deeplearning/dataset_on_titan-pascal/topdown_urban/4-27-rotated/meta/'
destination_path = '/home/mo/data_lifeng/sampled_training_data_5-4/chunks/'

if not os.path.exists(destination_path):
  os.mkdir(destination_path)

drive_ids = [i for i in os.listdir(source_path) if i[:2]=='HT']
for drive in drive_ids:
  #look for chunk file
  filelist = [i for i in os.listdir(source_path+drive) if i.endswith('.csv') and i[:-4].isdigit()]
  if len(filelist)>1:
    print('found more than one chunk file! exiting now')
    exit()
  else:
    chunk_file = filelist[0]
  copyfile(source_path+drive+'/'+chunk_file,destination_path+drive+'.csv')

