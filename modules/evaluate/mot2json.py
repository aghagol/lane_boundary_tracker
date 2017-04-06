#!/usr/bin/env python
""" 
This is a script for converting gt and tracks (in MOT format) to JSONs
"""
print(__doc__)

import sys, os
import numpy as np
import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input",help="list of sequences in a CSV file")
parser.add_argument("--output",help="output path to save JSONs")
args = parser.parse_args()

seqs = pd.read_csv(args.input)
for seq_idx,seq in seqs.iterrows():
  print('Working on sequence %s'%seqs.name[seq_idx])
  os.makedirs('%s/%s'%(args.output,seqs.name[seq_idx]))

  #process groundtruth
  track_file = seq.gpath
  print("Processing groundtruth: file located at %s"%(track_file))
  track = np.loadtxt(track_file,delimiter=',')
  out = {"frames":[],"class":"video"}
  frame_id_list = sorted(list(set(track[:,0])))
  print('...there is a total of %d frames'%(len(frame_id_list)))
  for frame_id in frame_id_list:
    tmp = {
        "timestamp": frame_id,
        "num": frame_id,
        "class": "frame",
        "annotations": [],
    }
    track_active = track[track[:,0]==frame_id,:]
    for point_id in range(track_active.shape[0]):
      points = {
        "dco": True,
        "id":     track_active[point_id,1],
        "x":      track_active[point_id,2],
        "y":      track_active[point_id,3],
        "height": track_active[point_id,4],
        "width":  track_active[point_id,5],
      }
      tmp["annotations"].append((points))
    out["frames"].append((tmp))
  jsonfile_path = '%s/%s/groundtruth.json'%(args.output,seqs.name[seq_idx])
  print("...writing to %s"%(jsonfile_path))
  with open(jsonfile_path,'w') as fjson:
    json.dump([out],fjson,indent=4)

  #process hypotheses
  track_file = seq.tpath
  print("Processing hypotheses: file located at %s"%(track_file))
  track = np.loadtxt(track_file,delimiter=',')
  out = {"frames":[],"class":"video"}
  frame_id_list = sorted(list(set(track[:,0])))
  print('...there is a total of %d frames'%(len(frame_id_list)))
  for frame_id in frame_id_list:
    tmp = {
        "timestamp": frame_id,
        "num": frame_id,
        "class": "frame",
        "hypotheses": [],
    }
    track_active = track[track[:,0]==frame_id,:]
    for point_id in range(track_active.shape[0]):
      points = {
        "id":     track_active[point_id,1],
        "x":      track_active[point_id,2],
        "y":      track_active[point_id,3],
        "height": track_active[point_id,4],
        "width":  track_active[point_id,5],
      }
      tmp["hypotheses"].append((points))
    out["frames"].append((tmp))
  jsonfile_path = '%s/%s/hypotheses.json'%(args.output,seqs.name[seq_idx])
  print("...writing to %s"%(jsonfile_path))
  with open(jsonfile_path,'w') as fjson:
    json.dump([out],fjson,indent=4)
