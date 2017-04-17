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

#this is an optional feature
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="a CSV file with 5 columns: \
  name (sequence name), dpath (path to detections CSV), tpath (path to tracking results CSV), \
  mpath (path to frame timestamps CSV) and gpath (path to groundtruth CSV).")
parser.add_argument("--delay",type=float,default=.01,help="delay in seconds for each frame.")
parser.add_argument("--margin",type=int,default=0,help="add this many pixels to plot margins.")
parser.add_argument("--groundtruth",action='store_true',help="Show ground-truth \
  instead of computer tracks.")
parser.add_argument("--fixed-axes",action='store_true',help="Use fixed axes for display.")
parser.add_argument("--window-size",type=float,default=100.,help="Display window size.")

args = parser.parse_args()
fixed_axes = args.fixed_axes
w = args.window_size
delay = args.delay

seqs = pd.read_csv(args.input)

fig, ax = plt.subplots(1,1,figsize=(6,6))
colors = np.random.rand(711,3) # create random colors for tracks

for seq_idx,seq in seqs.iterrows():

  print('Working on sequence %s'%seqs.name[seq_idx])

  dets = np.loadtxt(seq.dpath,delimiter=',')[:,:6]

  if os.path.exists(seq.mpath):
    timestamps = np.loadtxt(seq.mpath,delimiter=',')
  else:
    timestamps = np.zeros((dets.shape[0],2))

  if args.groundtruth:
    if not os.path.exists(seq.gpath): exit("GT data not there!")
    trks = np.loadtxt(seq.gpath,delimiter=',')
  else:
    if not os.path.exists(seq.tpath):
      print("No tracks file was found!")
      trks = np.zeros((1,7))
    else:
      trks = np.loadtxt(seq.tpath,delimiter=',')[:,:6]

  n_frames = int(dets[:,0].max())
  frame = 1
  while frame < n_frames:

    try:
      dets_cur = dets[dets[:,0]==frame,2:4]
      if not dets_cur.any():
        frame +=1
        continue

      ax.cla()

      if fixed_axes:
        ax.set_xlim([dets[:,2].min()-args.margin,dets[:,2].max()+args.margin])
        ax.set_ylim([dets[:,3].min()-args.margin,dets[:,3].max()+args.margin])
      else:
        xlim_low = np.floor(np.median(dets_cur[:,0])/w)*w
        ylim_low = np.floor(np.median(dets_cur[:,1])/w)*w
        ax.set_xlim([xlim_low-args.margin,xlim_low+w+args.margin])
        ax.set_ylim([ylim_low-args.margin,ylim_low+w+args.margin])

      # plot the detections as filled dots
      det_tail_color={1:'.5',2:'.65',3:'.8',4:'.9'}
      for fr in range(max(frame-4,0),frame): #tail
        ax.plot(dets[dets[:,0]==fr,2],dets[dets[:,0]==fr,3],'o',color=det_tail_color[frame-fr])
      ax.plot(dets_cur[:,0],dets_cur[:,1],'o',color='k')

      # get active tracks and plot them
      trks_active = trks[trks[:,0]==frame,:]
      for trk_active in trks_active:
        trk_idid = int(trk_active[1])
        trk_tail = trks[np.logical_and(trks[:,1]==trk_idid,np.logical_and(trks[:,0]<=frame,trks[:,0]>frame-1000)),:]
        ax.plot(trk_tail[:,2],trk_tail[:,3],color=colors[trk_idid%711,:])

      ax.set_title('frame %05d/%05d, time=%d'%(frame+1,n_frames,timestamps[frame,1]))
      plt.pause(delay)

      frame +=1

    except KeyboardInterrupt:
      print('')
      print('=================================================')
      print('...Enter j[int] to jump to frame')
      print('...Enter w[float] to adjust window width')
      print('...Enter d[float] to adjust delay')
      print('...Enter f to toggle display axes fit')
      print('...Enter q to quit')
      print('...Enter s to skip sequence and continue with next')
      inp = raw_input("...or just press Enter to resume: ")
      print('=================================================')
      if len(inp.strip()):
        if inp[0]=='j': frame = int(inp[1:])-1
        if inp[0]=='w': w = float(inp[1:])
        if inp[0]=='d': delay = float(inp[1:])
        if inp=='f': fixed_axes = not fixed_axes
        if inp=='q': exit('')
        if inp=='s': break

