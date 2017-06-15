#!/usr/bin/env python
""" 
This is a script for plotting detections and tracks (in MOT format)
CTRL+C to pause
"""
print(__doc__)

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from jsmin import jsmin

import postprocessing_util

#this is an optional feature
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to a CSV file containing sequence-related paths")
parser.add_argument("--output",help="output path to save tracklet fusion results")
parser.add_argument("--config",help="configuration JSON file")
args = parser.parse_args()

fmt = ['%05d','%05d','%011.5f','%011.5f','%05d','%04.2f']
with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["postprocess"]

if not os.path.exists(args.output):
  os.makedirs(args.output)

seqs = pd.read_csv(args.input)
for seq_idx,seq in seqs.iterrows():
  if os.path.exists('%s/%s.txt'%(args.output,seqs.name[seq_idx])): continue

  print('Working on sequence %s'%seqs.name[seq_idx])

  if not param['stitch']:
    print('\tStitching disabled; linking to tracker output!')
    os.system('ln -s %s %s'%(seq.tpath,'%s/%s.txt'%(args.output,seqs.name[seq_idx])))
    continue

  #read sequence data (detections, tracks, ...)
  # dets = np.loadtxt(seq.dpath,delimiter=',')
  # if os.path.exists(seq.mpath): timestamps = np.loadtxt(seq.mpath,delimiter=',')
  trks = np.loadtxt(seq.tpath,delimiter=',')

  out = postprocessing_util.fuse(trks,param)

  np.savetxt('%s/%s.txt'%(args.output,seqs.name[seq_idx]),out,fmt=fmt,delimiter=',')


