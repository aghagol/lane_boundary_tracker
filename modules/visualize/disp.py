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
import json

#suppress matplotlib warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="a CSV file with 5 columns: \
  name (sequence name), dpath (path to detections), tpath (path to tracking results), \
  mpath (path to frame timestamps) and gpath (path to groundtruth - optional).")
parser.add_argument("--delay",type=float,default=.01,help="delay in seconds for each frame.")
parser.add_argument("--margin",type=int,default=0,help="add this many pixels to plot margins.")
parser.add_argument("--groundtruth",action='store_true',help="Show ground-truth.")
parser.add_argument("--window-size",type=float,default=100.,help="Display window size.")
parser.add_argument("--config",help="configuration JSON file")

args = parser.parse_args()
w = args.window_size
delay = args.delay

with open(args.config) as fparam:
  param = json.load(fparam)["visualize"]

#over-ride paramaters from config file
if 'delay' in param: delay = param['delay']
if 'window_size' in param: w = param['window_size']

seqs = pd.read_csv(args.input)

fig, ax = plt.subplots(1,1,figsize=(9,9))
colors = np.random.rand(711,3) # create random colors for tracks
frame_buffer_size = 100

for seq_idx,seq in seqs.iterrows():

  print('Working on sequence %s'%seqs.name[seq_idx])

  dets = np.loadtxt(seq.dpath,delimiter=',')
  dets = dets[dets[:,6]>0,:] #remove the guide
  frame_timestamps = dict(zip(dets[:,0],dets[:,7]))

  if args.groundtruth:
    if not os.path.exists(seq.gpath): exit("GT data not there!")
    trks = np.loadtxt(seq.gpath,delimiter=',')
  elif param['show_tracks']:
    if not os.path.exists(seq.tpath): exit("\nNo tracks file was found!\n")
    trks = np.loadtxt(seq.tpath,delimiter=',')[:,:6]
    trks = trks[trks[:,4]>0,:] #remove the predictions with no matches

  frames = sorted(frame_timestamps)
  for frame_idx,frame in enumerate(frames):

    try:
      dets_cur = dets[dets[:,0]==frame,:]
      ax.cla()
      xlim_low = np.floor(np.median(dets_cur[:,2])/w)*w
      ylim_low = np.floor(np.median(dets_cur[:,3])/w)*w
      ax.set_xlim([xlim_low-args.margin,xlim_low+w+args.margin])
      ax.set_ylim([ylim_low-args.margin,ylim_low+w+args.margin])

      # plot detections
      for plot_frame_idx in range(max(frame_idx-frame_buffer_size,0),frame_idx):
        ax.plot(dets[dets[:,0]==frames[plot_frame_idx],2],dets[dets[:,0]==frames[plot_frame_idx],3],'o',color='k')

      # plot tracks
      start_frame = frames[max(frame_idx-frame_buffer_size,0)]
      if param['show_tracks']:
        trks_active_set = set(trks[np.logical_and(trks[:,0]<frame,trks[:,0]>start_frame),1])
        for trk_curr_id in trks_active_set:
          trk_curr_tail = trks[trks[:,1]==trk_curr_id,:]
          if trk_curr_tail.shape[0]<param['min_track_length']: continue
          trk_curr_tail = trk_curr_tail[trk_curr_tail[:,0]<frame,:]
          trk_curr_tail = trk_curr_tail[trk_curr_tail[:,0]>start_frame,:]
          ax.plot(trk_curr_tail[:,2],trk_curr_tail[:,3],color=colors[trk_curr_id%711,:])

      ax.set_title('frame number %05d/%05d, time=%d'%(frame_idx+1,len(frames),frame_timestamps[frame]))
      plt.pause(delay)

    except KeyboardInterrupt:
      print('')
      print('=================================================')
      print('...Enter j[int] to jump to frame')
      print('...Enter w[float] to adjust window width')
      print('...Enter d[float] to adjust delay')
      print('...Enter q to quit')
      print('...Enter s to skip sequence and continue with next')
      inp = raw_input("...or just press Enter to resume: ")
      print('=================================================')
      if len(inp.strip()):
        if inp[0]=='j': frame = int(inp[1:])-1
        if inp[0]=='w': w = float(inp[1:])
        if inp[0]=='d': delay = float(inp[1:])
        if inp=='q': exit('')
        if inp=='s': break

