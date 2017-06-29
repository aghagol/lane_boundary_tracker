#!/usr/bin/env python
""" 
This is a script for fusing tracking output (in MOT format)
"""
print(__doc__)

import sys, os
import numpy as np
import pandas as pd
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="CSV file containing paths")
parser.add_argument("--chunks",help="path to chunks metadata")
parser.add_argument("--output",help="path to output folder")
parser.add_argument("--config",help="configuration JSON file")

args = parser.parse_args()
output_path = args.output+'/'
chunks_path = args.chunks+'/'

if not os.path.exists(output_path):
  os.makedirs(output_path)

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["fuse"]

if not param['enable']: exit('Fusion is disabled. Aborting!')

seqs = pd.read_csv(args.input)

fmt = ['%.10f','%.10f','%.10f','%d']
for seq_idx,seq in seqs.iterrows():

  subdrive = seqs.name[seq_idx]
  drive = '_'.join(subdrive.split('_')[:2])
  print('Working on sequence %s'%subdrive)

  if not os.path.exists(output_path+subdrive):
    os.makedirs(output_path+subdrive)

  chunk_id_path = chunks_path+drive+'.csv'
  chunk_id = pd.read_csv(chunk_id_path)
  chunk_id.rename(columns=lambda x: x.strip(),inplace=True) #remove whitespace from headers

  tlla = np.loadtxt(seq.dpath,delimiter=',') #format: detection_id, timestamp, latitude, longitude, altitude, -1 (ground-truth label)
  tmap = {row[0]:row[1] for row in np.loadtxt(seq.tmap, delimiter=',')} #mapping from fake timestamps to true timestamps
  trks = np.loadtxt(seq.tpath,delimiter=',') #format: frame_id, target_id, x, y, detection_id, confidence
  trks = trks[trks[:,4]>0,:] #remove the guide

  for lb_number,target_id in enumerate(sorted(set(trks[:,1]))):

    output_fuse_chunk = output_path+'%s/%d_laneMarking.fuse'%(subdrive,lb_number)
    if os.path.exists(output_fuse_chunk): continue

    dets_ids = trks[trks[:,1]==target_id,4].astype(int).tolist()

    #prune short lane boundaries
    if len(dets_ids)<param['min_seq_length']: continue

    #make LLAT fuse (latitude, longitude, altitude, timestamp)
    out_fuse = []
    for det_id in dets_ids:
      out_fuse.append(tlla[tlla[:,0]==det_id,[2,3,4,1]].reshape(1,-1))
    out_fuse = np.vstack(out_fuse)
    
    #replace timestamp with chunk number
    in_chunk = np.zeros((out_fuse.shape[0]),dtype=bool)
    for row in range(out_fuse.shape[0]):
      timestamp = tmap[out_fuse[row,3]]
      mask = np.logical_and(chunk_id['StartTime']<=timestamp,chunk_id['EndTime']>timestamp)
      if mask.sum()==1: #ignore points that do not belong to any chunk (or belong to multiple chunks -> must not happen)
        in_chunk[row] = True
        out_fuse[row,3] = chunk_id['ChunkId'][mask]

    with open(output_fuse_chunk,'w') as fout:
      np.savetxt(fout,out_fuse[in_chunk,:],fmt=fmt)
