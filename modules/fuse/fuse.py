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

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="CSV file containing paths")
parser.add_argument("--input-root", help="path to dataset root")
parser.add_argument("--output",help="path to output folder")
parser.add_argument("--config",help="configuration JSON file")

args = parser.parse_args()
output_path = args.output+'/'
root_path = args.input_root+'/'

os.makedirs(output_path)

with open(args.config) as fparam:
  param = json.load(fparam)["fuse"]

seqs = pd.read_csv(args.input)

fmt = ['%.10f','%.10f','%.10f','%d']
for seq_idx,seq in seqs.iterrows():

  drive = seqs.name[seq_idx]
  print('Working on sequence %s'%drive)

  chunk_id_path = [csv_file for csv_file in os.listdir(root_path+drive) if csv_file.endswith('_image.csv')]
  assert len(chunk_id_path)==1, 'ERROR: found %d pose files in %s'%(len(chunk_id_path),root_path+drive)
  chunk_id_path = root_path+drive+'/'+chunk_id_path[0]
  chunk_id = pd.read_csv(chunk_id_path)
  chunk_id.rename(columns=lambda x: x.strip(),inplace=True) #remove whitespace from headers

  dets = np.loadtxt(seq.dpath,delimiter=',')
  trks = np.loadtxt(seq.tpath,delimiter=',')
  trks = trks[trks[:,4]>0,:] #remove the guide

  for lb_number,target_id in enumerate(sorted(set(trks[:,1]))):
    dets_ids = trks[trks[:,1]==target_id,4].astype(int).tolist()
    out_fuse = []
    for det_id in dets_ids:
      out_fuse.append(dets[det_id-1,[2,3,4,1]].reshape(1,-1))
    out_fuse = np.vstack(out_fuse)

    output_fuse = '%d_laneMarking_l2polyline.fuse'%(lb_number)
    with open(output_path+output_fuse,'w') as fout:
      np.savetxt(fout,out_fuse,fmt=fmt)

    for row in range(out_fuse.shape[0]):
      timestamp = out_fuse[row,3]
      out_fuse[row,3] = chunk_id['ChunkId'][np.logical_and(chunk_id['StartTime']<=timestamp,chunk_id['EndTime']>timestamp)]
    output_fuse = '%d_laneMarking.fuse'%(lb_number)
    with open(output_path+output_fuse,'w') as fout:
      np.savetxt(fout,out_fuse,fmt=fmt)
