#!/usr/bin/env python
""" 
This is a script for merging fuse files from drive segments for each drive
lane marking outputs from previous stage must be grouped according to drive IDs
"""

import sys, os
from shutil import copyfile
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--input",      help="path to fuse output folder")
parser.add_argument("--output",     help="path to merged output folder")
parser.add_argument("--config",     help="configuration JSON file")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

if args.verbosity>=2:
  print(__doc__)

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["fuse"]

if not param['merge_subdrives']:
  if args.verbosity>=1:
    print('Subdrive merging is disabled. Skipping.')
  exit()

if not os.path.exists(args.output):
  os.makedirs(args.output)

#get a list of sequence names
fuse_dir_list = [i for i in os.listdir(args.input) if os.path.isdir(args.input+'/'+i)]

#compute a mapping between sequence names and drive IDs
#note: sequence names have the structure: driveID_part_number
fuse_to_drive_map = {fuse_dir:'_'.join(fuse_dir.split('_')[:2]) for fuse_dir in fuse_dir_list}

#for each drive, create a new folder and put its lane markings there
drive_list = sorted(set(fuse_to_drive_map.values()))
for drive in drive_list:
  if not os.path.exists(args.output+'/'+drive):
    os.makedirs(args.output+'/'+drive)

#copy fuse files into respective drive_id folders
#note: fuse files must also be renamed because two sequences within the same drive can have 
#fuse files with same names while corresponding to different lane markings
line_counter = {drive:0 for drive in drive_list}
for fuse_dir in fuse_dir_list:
  fuse_files = [i for i in os.listdir(args.input+'/'+fuse_dir) if i.endswith('_laneMarking.fuse')]
  for fuse_file in fuse_files:
    output_file = args.output+'/'+fuse_to_drive_map[fuse_dir]+'/%d_laneMarking.fuse'%(line_counter[fuse_to_drive_map[fuse_dir]])
    if not os.path.exists(output_file):
      copyfile(args.input+'/'+fuse_dir+'/'+fuse_file,output_file)
    line_counter[fuse_to_drive_map[fuse_dir]] +=1

#print some useful info
for drive in line_counter:
  if args.verbosity>=2:
    print('found %d boundaries in drive %s'%(line_counter[drive],drive))
