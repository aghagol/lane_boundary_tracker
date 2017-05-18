#!/usr/bin/env python
"""
This script creates a drives list file (if it does not exist)
"""
print(__doc__)

import os,sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input",help="path to input annotations")
parser.add_argument("--drives",help="path to drives list file")
args = parser.parse_args()

if not os.path.exists(args.drives):
  print("WARNING: generating a drive list because it is missing")
  image_names = [i[:-8] for i in os.listdir(args.input) if i.endswith('.png.fuse')]
  print('found %d annotated images in %s'%(len(image_names),args.input))
  drive_names = sorted(set(['_'.join(image_name.split('_')[:2]) for image_name in image_names]))
  drive_sizes = [len([image_name for image_name in image_names if '_'.join(image_name.split('_')[:2])==drive_name]) for drive_name in drive_names]
  drive_names_sorted = [drive_name for (drive_size,drive_name) in sorted(zip(drive_sizes,drive_names))][::-1]
  drive_sizes_sorted = sorted(drive_sizes)[::-1]
  with open(args.drives,'w') as fdrivelist:
    for i,drive_name in enumerate(drive_names_sorted):
      print('...%d images in drive %s'%(drive_sizes_sorted[i],drive_name))
      fdrivelist.write('%s\n'%(drive_name))