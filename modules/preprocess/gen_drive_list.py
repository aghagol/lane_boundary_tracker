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
  image_list = [i[:-8] for i in os.listdir(args.input) if i.endswith('.png.fuse')]
  print('found %d annotated images in %s'%(len(image_list),args.input))
  drive_list = sorted(set(['_'.join(imagename.split('_')[:2]) for imagename in image_list]))
  with open(args.drives,'w') as fdrivelist:
    for drive in drive_list:
      fdrivelist.write('%s\n'%(drive))