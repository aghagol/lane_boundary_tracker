#!/usr/bin/env python
"""
This script creates a drives list file if it does not exist already
"""

import os,sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--images",     help="path to FCN output")
parser.add_argument("--drives",     help="path to drives list file")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

if args.verbosity>=2:
  print(__doc__)

if not os.path.exists(args.drives):
  if args.verbosity>=2:
    print("Generating a drive list because it is missing")

  #Assuming the following data hierarchy:
  #
  #images/drive_id/startTimestamp_endTimestamp_pred.png

  drive_names = [i for i in os.listdir(args.images) if i[:2]=='HT']
  with open(args.drives,'w') as fdrivelist:
    for drive_name in drive_names:
      fdrivelist.write('%s\n'%(drive_name))