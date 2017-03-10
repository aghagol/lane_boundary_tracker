""" 
This is a script for plotting detections and tracks
Note: detections and tracks must be provided in the MOT format
CTRL+C to pause
"""
print(__doc__)

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_seqs", help="a CSV file with 3 columns: \
  name (sequence name), dpath (path to detections CSV) and tpath (path to tracking results CSV).")
parser.add_argument("-d","--delay",type=float,default=.01,help="delay in seconds for each frame.")
parser.add_argument("--margin",type=int,default=50,help="add this many pixels to plot margins.")
args = parser.parse_args()

seqs = pd.read_csv(args.input_seqs)

fig, ax = plt.subplots(1,1)
colors = np.random.rand(711,3) # create random colors for tracks

for seq_idx,seq in seqs.iterrows():

  print('Working on sequence %s'%seqs.name[seq_idx])

  dets = np.loadtxt(seq.dpath,delimiter=',')[:,:6]
  trks = np.loadtxt(seq.tpath,delimiter=',')[:,:6]

  # center
  dets[:,2] += dets[:,4]/2.
  dets[:,3] += dets[:,5]/2.
  trks[:,2] += trks[:,4]/2.
  trks[:,3] += trks[:,5]/2.

  for frame in range(int(trks[:,0].max())-1):

    try:
      ax.cla()
      ax.set_xlim([dets[:,2].min()-args.margin,dets[:,2].max()+args.margin])
      ax.set_ylim([dets[:,3].min()-args.margin,dets[:,3].max()+args.margin])

      # plot the detections as filled dots
      ax.plot(dets[dets[:,0]==frame,2],dets[dets[:,0]==frame,3],'ko')

      # get active tracks and plot them
      trks_active = trks[trks[:,0]==frame,:]
      for trk_alive in trks_active:
        trk_idid = int(trk_alive[1])
        trk_tail = trks[np.logical_and(trks[:,1]==trk_idid,trks[:,0]<=frame),:]
        ax.plot(trk_tail[:,2],trk_tail[:,3],'r',color=colors[trk_idid%711,:])

      ax.set_title('frame %d'%frame)
      plt.pause(args.delay)

    except KeyboardInterrupt:
      inp = raw_input(" Enter 'q' to quit, 's' to skip: ")
      if inp=='q': exit('')
      if inp=='s': break

