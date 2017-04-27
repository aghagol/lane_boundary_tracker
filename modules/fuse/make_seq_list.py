#!/usr/bin/env python
""" 
Creating a list of sequences as input to disp.py

output CSV has 5 columns:
  name..............this is the sequence name
  dpath.............this is the detection file path
  mpath.............this is the frame timestamps file path (if any)
  tpath.............this is the track file path
  gpath.............this is the groundtruth file path
"""
print(__doc__)

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to MOT dataset")
parser.add_argument("--tracks",help="path to predicted tracks")
parser.add_argument("--output",help="output path to save seq.csv")
args = parser.parse_args()

sequences = sorted([i for i in os.listdir(args.input)])
if os.path.exists(args.tracks):
	seq_set_trks = set([i[:-4] for i in os.listdir(args.tracks) if i.endswith('txt')])
	seq_set_dets = set(sequences)
	for seq in sequences:
	    if not seq in seq_set_trks:
	        seq_set_dets.remove(seq)
	sequences = sorted(list(seq_set_dets))

seqs = {}
seqs['name'] = sequences
seqs['dpath'] = ['%s/%s/det/tlla.txt'%(args.input,seq) for seq in sequences]
seqs['tpath'] = ['%s/%s.txt'%(args.tracks,seq) for seq in sequences]

seqs_df = pd.DataFrame(seqs)
seqs_df = seqs_df[['name','dpath','tpath']]

seqs_df.to_csv(args.output,index=False,header=True)

