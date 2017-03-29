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

parser = argparse.ArgumentParser()
parser.add_argument("input_seqs", help="a CSV file with 3 columns: \
  name (sequence name), dpath (path to detections CSV) and tpath (path to tracking results CSV).")
parser.add_argument("-d","--delay",type=float,default=.01,help="delay in seconds for each frame.")
parser.add_argument("--margin",type=int,default=50,help="add this many pixels to plot margins.")
args = parser.parse_args()

seqs = pd.read_csv(args.input_seqs)

fig, ax = plt.subplots(1,1,figsize=(9,9))
colors = np.random.rand(711,3) # create random colors for tracks

w = 100. #display windows size
for seq_idx,seq in seqs.iterrows():

  print('Working on sequence %s'%seqs.name[seq_idx])

  dets = np.loadtxt(seq.dpath,delimiter=',')[:,:6]
  trks = np.loadtxt(seq.tpath,delimiter=',')[:,:6]

  # center
  dets[:,2] += dets[:,4]/2.
  dets[:,3] += dets[:,5]/2.
  trks[:,2] += trks[:,4]/2.
  trks[:,3] += trks[:,5]/2.

  n_frames = int(dets[:,0].max())
  frame = 1
  while frame < n_frames:

    try:
      dets_cur = dets[dets[:,0]==frame,2:4]
      if not dets_cur.any():
        continue

      ax.cla()

      # ax.set_xlim([dets[:,2].min()-args.margin,dets[:,2].max()+args.margin])
      # ax.set_ylim([dets[:,3].min()-args.margin,dets[:,3].max()+args.margin])

      xlim_low = np.floor(np.median(dets_cur[:,0])/w)*w
      ylim_low = np.floor(np.median(dets_cur[:,1])/w)*w
      ax.set_xlim([xlim_low-w,xlim_low+w])
      ax.set_ylim([ylim_low-w,ylim_low+w])

      # plot the detections as filled dots
      for fr in range(max(frame-1,0),frame): #tail
        ax.plot(dets[dets[:,0]==fr,2],dets[dets[:,0]==fr,3],'r.')
      ax.plot(dets_cur[:,0],dets_cur[:,1],'k.')

      # get active tracks and plot them
      trks_active = trks[trks[:,0]==frame,:]
      for trk_active in trks_active:
        trk_idid = int(trk_active[1])
        trk_tail = trks[np.logical_and(trks[:,1]==trk_idid,np.logical_and(trks[:,0]<=frame,trks[:,0]>frame-1000)),:]
        ax.plot(trk_tail[:,2],trk_tail[:,3],color=colors[trk_idid%711,:])

      ax.set_title('frame %d (out of %d)'%(frame+1,n_frames))
      plt.pause(args.delay)

    except KeyboardInterrupt:
      os.system('clear')
      print('Menu:')
      print('\tj\tjump to frame')
      print('\tw\tadjust window width')
      print('\tq\tquit')
      print('\ts\tskip sequence and continue with next')
      inp = raw_input("\nPress Enter to resume: ")
      if len(inp.strip()):
        if inp[0]=='j': frame=int(inp[1:])-1
        if inp[0]=='w': w=float(inp[1:])
        if inp=='q': exit('')
        if inp=='s': break
      else:
        os.system('clear')

    frame +=1

