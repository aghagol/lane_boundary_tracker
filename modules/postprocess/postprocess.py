#!/usr/bin/env python
""" 
This is a script for performing a sequece of postprocessing operations on the tracking resutls
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

out_fmt = ['%05d','%05d','%011.5f','%011.5f','%05d','%04.2f'] #frame_id, target_id, x, y, detection_id, confidence
with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["postprocess"]
flag_stitch = param['stitch_tracklets']
flag_reduce = param['point_reduction']
flag_postprocess = flag_stitch or flag_reduce

if not os.path.exists(args.output):
  os.makedirs(args.output)

seqs = pd.read_csv(args.input)

if not flag_postprocess:
  print('No post-processing required; linking to tracker output.')
  for seq_idx,seq in seqs.iterrows():
    os.system('ln -s %s %s'%(seq.tpath,'%s/%s.txt'%(args.output,seqs.name[seq_idx])))

for seq_idx,seq in seqs.iterrows():
  output_path_final = '%s/%s.txt'%(args.output,seqs.name[seq_idx])
  if os.path.exists(output_path_final): continue

  print('Working on sequence %s'%seqs.name[seq_idx])
  
  #read detections (if needed)
  # dets = np.loadtxt(seq.dpath,delimiter=',')

  #start with the original tracking results
  trks = np.loadtxt(seq.tpath,delimiter=',')

  if flag_stitch:
    trks = postprocessing_util.stitch(trks,param)

  if flag_reduce:
    trks = postprocessing_util.reducer(trks,param)

  np.savetxt(output_path_final,trks,fmt=out_fmt,delimiter=',')


