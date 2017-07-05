#!/usr/bin/env python
"""
This script creates a drives list file (if it does not exist)
"""
print(__doc__)

import os,sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--images",help="path to FCN output")
parser.add_argument("--drives",help="path to drives list file")
args = parser.parse_args()

if not os.path.exists(args.drives):
  print("WARNING: generating a drive list because it is missing")
  drive_names = [i[:-8] for i in os.listdir(args.images) if i[:2]=='HT']
  with open(args.drives,'w') as fdrivelist:
    for drive_name in drive_names:
      fdrivelist.write('%s\n'%(drive_name))