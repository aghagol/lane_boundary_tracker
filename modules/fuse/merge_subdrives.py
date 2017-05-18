#!/usr/bin/env python
""" 
This is a script for merging fuse files from drive segments for each drive
"""
print(__doc__)

import sys, os
from shutil import copyfile
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to fuse output folder")
parser.add_argument("--output",help="path to merged output folder")
parser.add_argument("--config",help="configuration JSON file")

args = parser.parse_args()
input_path = args.input+'/'
output_path = args.output+'/'

os.makedirs(output_path)

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["fuse"]
if not param['enable']: exit()

fuse_dir_list = [i for i in os.listdir(input_path) if os.path.isdir(input_path+i)]
fuse_to_drive_map = {fuse_dir:'_'.join(fuse_dir.split('_')[:2]) for fuse_dir in fuse_dir_list}
drive_list = sorted(set(fuse_to_drive_map.values()))

for drive in drive_list:
  os.makedirs(output_path+drive)

line_counter = {drive:0 for drive in drive_list}
for fuse_dir in fuse_dir_list:
  fuse_files = [i for i in os.listdir(input_path+fuse_dir) if i.endswith('_laneMarking.fuse')]
  for fuse_file in fuse_files:
    name = '%d_laneMarking.fuse'%(line_counter[fuse_to_drive_map[fuse_dir]])
    copyfile(input_path+fuse_dir+'/'+fuse_file,output_path+fuse_to_drive_map[fuse_dir]+'/'+name)
    line_counter[fuse_to_drive_map[fuse_dir]] +=1

for drive in line_counter:
  print('found %d boundaries in drive %s'%(line_counter[drive],drive))
