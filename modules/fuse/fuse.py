#!/usr/bin/env python
""" 
This is a script for fusing tracking output (in MOT format)
"""

import sys, os
import numpy as np
import pandas as pd
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--input",      help="CSV file containing paths")
parser.add_argument("--chunks",     help="path to chunks metadata")
parser.add_argument("--output",     help="path to output folder")
parser.add_argument("--config",     help="configuration JSON file")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

if args.verbosity>=2:
  print(__doc__)

if not os.path.exists(args.output):
  os.makedirs(args.output)

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["fuse"]

if not param['enable']: exit('Fusion is disabled. Aborting!')

seqs = pd.read_csv(args.input)

fmt = ['%.10f','%.10f','%.10f','%d']
for seq_idx,seq in seqs.iterrows():

  subdrive = seqs.name[seq_idx]
  drive = '_'.join(subdrive.split('_')[:2])

  if args.verbosity>=2:
    print('Working on sequence %s'%subdrive)

  if not os.path.exists(args.output+'/'+subdrive):
    os.makedirs(args.output+'/'+subdrive)

  chunk_id_path = args.chunks+'/'+drive+'.csv'
  chunk_id = pd.read_csv(chunk_id_path)
  chunk_id.rename(columns=lambda x: x.strip(),inplace=True) #remove whitespace from headers

  tlla = np.loadtxt(seq.dpath,delimiter=',') #format: detection_id, timestamp, latitude, longitude, altitude, -1 (ground-truth label)
  tmap = {row[0]:row[1] for row in np.loadtxt(seq.tmap, delimiter=',')} #mapping from fake timestamps to true timestamps
  trks = np.loadtxt(seq.tpath,delimiter=',') #format: frame_id, target_id, x, y, detection_id, confidence
  trks = trks[trks[:,4]>0,:] #remove the guide

  for lb_number,target_id in enumerate(sorted(set(trks[:,1]))):

    output_fuse_chunk = args.output+'/%s/%d_laneMarking.fuse'%(subdrive,lb_number)
    if os.path.exists(output_fuse_chunk): continue

    trk_active = trks[trks[:,1]==target_id,:]
    dets_ids = trk_active[:,4].astype(int).tolist()

    #prune short lane boundaries (polylines)
    flag_too_few_points = len(dets_ids)<param['min_seq_length']

    #prune tracks that lie in a bbox of size smaller than min_bbox_size
    bbox_x = trk_active[:,2].max()-trk_active[:,2].min() #width
    bbox_y = trk_active[:,3].max()-trk_active[:,3].min() #height
    flag_too_small = max(bbox_x,bbox_y)<param['min_bbox_size']

    flag_skip_track = flag_too_few_points or flag_too_small

    if flag_skip_track: continue

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
