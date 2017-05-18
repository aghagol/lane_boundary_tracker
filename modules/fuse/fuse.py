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
  os.makedirs(output_path+subdrive)

  chunk_id_path = chunks_path+drive+'.csv'
  chunk_id = pd.read_csv(chunk_id_path)
  chunk_id.rename(columns=lambda x: x.strip(),inplace=True) #remove whitespace from headers

  dets = np.loadtxt(seq.dpath,delimiter=',')
  tmap = np.loadtxt(seq.tmap, delimiter=',')
  tmap = {row[0]:row[1] for row in tmap}
  trks = np.loadtxt(seq.tpath,delimiter=',')
  trks = trks[trks[:,4]>0,:] #remove the guide

  for lb_number,target_id in enumerate(sorted(set(trks[:,1]))):
    dets_ids = trks[trks[:,1]==target_id,4].astype(int).tolist()
    out_fuse = []
    for det_id in dets_ids:
      out_fuse.append(dets[dets[:,0]==det_id,[2,3,4,1]].reshape(1,-1)) #LLAT format
    out_fuse = np.vstack(out_fuse)

    output_fuse = '%s/%d_laneMarking_l2polyline.fuse'%(subdrive,lb_number)
    with open(output_path+output_fuse,'w') as fout:
      np.savetxt(fout,out_fuse,fmt=fmt)

    for row in range(out_fuse.shape[0]):
      timestamp = tmap[out_fuse[row,3]]
      mask = np.logical_and(chunk_id['StartTime']<=timestamp,chunk_id['EndTime']>timestamp)
      assert mask.sum()==1, 'Each detection must belong to one and only one chunk (%d)'%mask.sum()
      out_fuse[row,3] = chunk_id['ChunkId'][mask]
    output_fuse = '%s/%d_laneMarking.fuse'%(subdrive,lb_number)
    with open(output_path+output_fuse,'w') as fout:
      np.savetxt(fout,out_fuse,fmt=fmt)
