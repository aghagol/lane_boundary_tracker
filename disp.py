""" 
This is a script for plotting detections and tracks
Note: detections and tracks must be provided in the MOT format
CTRL+C to skip to the next sequence
"""
print(__doc__)

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_seqs", 
  help="a tab-separated CSV file with 3 columns... \
  name: seq name, dpath: path to detections and \
  tpath: path to tracking results.")
args = parser.parse_args()

seqs = pd.read_csv(args.input_seqs,sep='\t')

fig, ax = plt.subplots(1,1)
colors = np.random.rand(711,3) # create random colors for tracks
margin = 50 # add this many ypixels to plots' margins

for seq_idx,seq in seqs.iterrows():

  print('Working on sequence %s'%seq.name)

  try:
    dets = np.loadtxt(seq.dpath,delimiter=',')[:,:6]
    trks = np.loadtxt(seq.tpath,delimiter=',')[:,:6]

    # center
    dets[:,2] += dets[:,4]/2.
    dets[:,3] += dets[:,5]/2.
    trks[:,2] += trks[:,4]/2.
    trks[:,3] += trks[:,5]/2.
    
    for frame in range(int(trks[:,0].max())-1):
      ax.cla()
      ax.set_xlim([dets[:,2].min()-margin,dets[:,2].max()+margin])
      ax.set_ylim([dets[:,3].min()-margin,dets[:,3].max()+margin])

      # plot the detections as filled dots
      ax.plot(dets[dets[:,0]==frame,2],dets[dets[:,0]==frame,3],'ko')

      # get active tracks and plot them
      trks_active = trks[trks[:,0]==frame,:]
      for trk_alive in trks_active:
        trk_idid = int(trk_alive[1])
        trk_tail = trks[np.logical_and(trks[:,1]==trk_idid,trks[:,0]<=frame),:]
        ax.plot(trk_tail[:,2],trk_tail[:,3],'r',color=colors[trk_idid%711,:])

      ax.set_title('frame %d'%frame)
      plt.pause(.1)

  except KeyboardInterrupt:
    if raw_input("... Press Enter to continue ('q' to stop)...")=='q': exit('')
    continue





















