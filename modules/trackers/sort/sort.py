#!/usr/bin/env python
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
import json
from filterpy.kalman import KalmanFilter

def d2t_sim(z,HFx): #detection to track similarity
  """
  Computes similarity between a detection (2x1 numpy array) to a prediction (2x1 numpy array)
  """
  # return np.dot((z-HFx).T, np.dot(np.inverse(S),(z-HFx)))
  return np.exp(-np.sqrt(np.dot((z-HFx).T, (z-HFx))))

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects
  """
  count = 0
  def __init__(self,pos,initial_velocity=np.zeros((2,1))):
    """
    Initialises a tracker
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=4, dim_z=2)
    self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])

    # self.kf.R[2:,2:] *= 10. #default is 10
    # self.kf.P[2:,2:] *= 1000. #give high uncertainty to the unobservable initial velocities
    # self.kf.P *= 10. #default is 10
    # self.kf.Q[2:,2:] *= 0.01

    self.kf.x[:2] = pos[:2]
    self.kf.x[2:] = initial_velocity
    self.age_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.confidence = .1
    self.curr_det = pos.squeeze() #mohammad

  def update(self,pos):
    """
    Updates the state vector with observations
    """
    self.curr_det = pos.squeeze()
    self.age_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.confidence = 1.
    self.kf.update(pos[:2])

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()
    self.age += 1
    if(self.age_since_update>0):
      self.hit_streak = 0
    self.age_since_update += 1
    self.confidence *= .95
    # self.history.append(self.kf.x)
    # return self.history[-1]
    print(self.kf.x.shape)
    self.curr_det = np.concatenate([self.kf.x[:2].squeeze(),[0]])
    return self.kf.x

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x

def associate_detections_to_trackers(detections,trackers,d2t_dist_threshold):
  """
  Assigns detections (d x 2 numpy array) to tracked object (t x 2 numpy array)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,2),dtype=int)
  d2t_sim_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      d2t_sim_matrix[d,t] = d2t_sim(det.reshape(2,1),trk.reshape(2,1))
  matched_indices = linear_assignment(-d2t_sim_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low d2t_sim
  matches = []
  for m in matched_indices:
    if(d2t_sim_matrix[m[0],m[1]]<np.exp(-d2t_dist_threshold)):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):
  def __init__(self,
      max_age_since_update=1,
      min_hits=3,
      d2t_dist_threshold_tight=1,
      d2t_dist_threshold_loose=3,
    ):
    """
    Sets key parameters for SORT
    """
    self.max_age_since_update = max_age_since_update
    self.min_hits = min_hits
    self.d2t_dist_threshold_tight = d2t_dist_threshold_tight
    self.d2t_dist_threshold_loose = d2t_dist_threshold_loose
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,score],[x,y,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),2))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()
      trk[:] = [pos[0], pos[1]]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    #----------------------------drop d2t_sim_theshold when there are no existing trackers (mohammad)
    if len([trk for trk in self.trackers if trk.hits>0]):
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets[:,:2],trks,self.d2t_dist_threshold_tight)
    else:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets[:,:2],trks,self.d2t_dist_threshold_loose)
    #--------------------------------------------------------------------------------------------

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:].reshape(3,1))

    #create and initialise new trackers for unmatched detections
    #---------------but first compute average motion for track initialization (mohammad's add-on)
    tracker_states = [np.array(trk.kf.x) for trk in self.trackers if trk.hits>0]
    if len(tracker_states):
      avg_velocity = np.array(tracker_states).mean(axis=0)[2:].reshape(2,1)
    else:
      avg_velocity = np.zeros((2,1))
    #--------------------------------------------------------------------------------------------
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:].reshape(3,1),avg_velocity)
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        # d = trk.get_state().squeeze()
        d = trk.curr_det
        # if((trk.age_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
        if True: #mohammad: output all tracks except guides and only when associated with a detection
          ret.append(np.concatenate(([trk.id+1],d,[trk.confidence])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.age_since_update > self.max_age_since_update):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,2))

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='SORT demo')
  parser.add_argument("--input",help="path to MOT dataset")
  parser.add_argument("--output",help="output path to save tracking results")
  parser.add_argument("--config",help="configuration JSON file")
  args = parser.parse_args()
  total_time = 0.0
  total_frames = 0

  with open(args.config) as fparam:
    param = json.load(fparam)["sort"]
  print(param)

  os.makedirs(args.output)

  sequences = os.listdir(args.input)
  for seq in sequences:
    mot_tracker = Sort(
      max_age_since_update=param['max_age_after_last_update'],
      d2t_dist_threshold_tight=param['d2t_dist_threshold_tight'],
      d2t_dist_threshold_loose=param['d2t_dist_threshold_loose'],
    ) #create instance of the SORT tracker

    seq_dets = np.loadtxt('%s/%s/det/det.txt'%(args.input,seq),delimiter=',') #load detections

    out_file = open('%s/%s.txt'%(args.output,seq),'w')

    print("Processing %s"%(seq))
    for frame in range(int(seq_dets[:,0].max())):
      frame += 1 #detection and frame numbers begin at 1
      dets = seq_dets[seq_dets[:,0]==frame,:][:,[2,3,6]].reshape(-1,3)
      total_frames += 1

      start_time = time.time()
      trackers = mot_tracker.update(dets)
      cycle_time = time.time() - start_time
      total_time += cycle_time

      for d in trackers:
        print('%05d,%05d,%011.5f,%011.5f,%05d,%04.2f'%(frame,d[0],d[1],d[2],d[3],d[4]),file=out_file)

    out_file.close()

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))



