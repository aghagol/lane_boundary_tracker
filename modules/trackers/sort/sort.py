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
# from numba import jit
import os
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import time
import argparse
import json
from jsmin import jsmin
from filterpy.kalman import KalmanFilter

def d2t_sim(z,x): #detection to track similarity
  """
  Computes similarity between a detection (2x1 numpy array) and a prediction (2x1 numpy array)
  """
  # return np.exp(-np.sqrt(np.dot((z-HFx).T, np.dot(np.linalg.inv(S),(z-HFx)))))
  # return np.exp(-np.sqrt(np.dot((z-HFx).T, (z-HFx))))

  y = z-(x[:2].reshape(2,1)) #vector from detection to prediction (displacement)
  v = x[2:].reshape(2,1) #velocity (motion) vector
  v_l2_squared = np.sum(v*v)
  if v_l2_squared>0:
    d = y-(np.sum(y*v)/v_l2_squared)*v #orthogonal projection of displacement onto motion
  else:
    d = y
  return np.exp(-np.sqrt(np.sum(d*d)))

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects
  """
  count = 0
  def __init__(self,initial_state=np.zeros((4,1)),det_idx=0,det_time=None,motion_model_var=1,observation_var=1):
    """
    Initialises a tracker
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=4, dim_z=2)
    self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])

    # self.kf.P *= 10. #initial state variance/uncertainty - default 10
    # self.kf.P[2:,2:] *= 1000. #initial motion variance/uncertainty - default 1000
    self.kf.Q[2:,2:] *= motion_model_var #model-induced motion variance - default 0.01
    self.kf.R *= observation_var #observation variance/uncertainty - default 10

    self.kf.x = initial_state
    self.age_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.confidence = .1
    self.ret = initial_state[:2]
    self.det_idx = det_idx
    self.det_time = det_time
    self.det_x = initial_state[:2]

  def update(self,target_location,det_idx=0,det_time=None):
    """
    Updates the state vector with observations
    """
    self.age_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.confidence = 1.
    self.kf.update(target_location)
    # self.ret = target_location
    self.ret = self.kf.x[:2]
    self.det_idx = det_idx
    self.det_time = det_time
    self.det_x = target_location

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
    self.ret = self.kf.x[:2]
    self.det_idx = 0
    return self.kf.x
    # self.history.append(self.kf.x)
    # return self.history[-1]

def associate_detections_to_trackers(detections,trackers,d2t_dist_thresh):
  """
  Assigns detections (d x 2 numpy array) to tracked object (t x 2 numpy array)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,2),dtype=int)
  d2t_sim_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      d2t_sim_matrix[d,t] = d2t_sim(det.reshape(2,1),trk.reshape(4,1))
  matched_indices = linear_assignment(1-d2t_sim_matrix)

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
    if(d2t_sim_matrix[m[0],m[1]]<np.exp(-d2t_dist_thresh[m[1]])):
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
      max_mov_since_update=1,
      min_hits=3,
      d2t_dist_threshold_tight=10,
      d2t_dist_threshold_loose=10,
      motion_model_variance=1,
      observation_variance=1,
      motion_init_pose=False,
      motion_init_sync=False,
    ):
    """
    Sets key parameters for SORT
    """
    self.trackers = []
    self.frame_count = 0
    self.time_gap = 0
    self.time_now = 0
    self.max_age_since_update = max_age_since_update
    self.max_mov_since_update = max_mov_since_update
    self.min_hits = min_hits
    self.d2t_dist_threshold_tight = d2t_dist_threshold_tight
    self.d2t_dist_threshold_loose = d2t_dist_threshold_loose
    self.motion_model_variance = motion_model_variance
    self.observation_variance = observation_variance
    self.motion_init_pose = motion_init_pose
    self.motion_init_sync = motion_init_sync

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,u,v,idx,time],[x,y,u,v,idx,time],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count +=1
    self.time_gap = dets[0,5]-self.time_now if self.frame_count>1 else 0
    self.time_now = dets[0,5]

    self.time_gap += 1e-7 #to prevent numerical instability

    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),4))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      self.trackers[t].kf.F = np.array([[1,0,self.time_gap,0],[0,1,0,self.time_gap],[0,0,1,0],[0,0,0,1]])
      self.trackers[t].kf.Q *= self.time_gap #more uncertainty for large time steps
      target_state = self.trackers[t].predict()
      trk[:] = target_state.squeeze()
      if(np.any(np.isnan(target_state))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in to_del[::-1]:
      self.trackers.pop(t)

    #associate detections with trackers
    #EDIT ------------ tighten d2t_dist_theshold after the first hit (mohammad)
    d2t_dist_thresh = []
    for t, trk in enumerate(self.trackers):
      if trk.hits>0:
        d2t_dist_thresh.append(self.d2t_dist_threshold_tight)
      else:
        d2t_dist_thresh.append(self.d2t_dist_threshold_loose)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets[:,:2],trks,d2t_dist_thresh)
    #-------------------------------------------------------

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:2].reshape(2,1),det_idx=dets[d,4],det_time=self.time_now)

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
      if self.motion_init_pose:
        initial_state = dets[i,:4].reshape(4,1)
      elif self.motion_init_sync:
        #compute average motion for track initialization
        tracker_states = [np.array(trk.kf.x) for trk in self.trackers if trk.hits>0]
        if len(tracker_states):
          avg_velocity = np.array(tracker_states).mean(axis=0)[2:].reshape(2,1)
        else:
          avg_velocity = np.zeros((2,1))
        initial_state = np.concatenate((dets[i,:2].reshape(2,1),avg_velocity))
      else:
        initial_state = np.concatenate((dets[i,:2].reshape(2,1),np.zeros((2,1))))
      trk = KalmanBoxTracker(
        initial_state=initial_state,
        det_idx=dets[i,4],
        det_time=self.time_now,
        motion_model_var=self.motion_model_variance,
        observation_var=self.observation_variance)
      self.trackers.append(trk)

    #output tracking results
    for trk in self.trackers:
      d = trk.ret.squeeze() #customized return value
      # if((trk.age_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
      if True: #EDIT -------- output all (mohammad)
        ret_item = []
        ret_item.append(trk.id+1) # +1 as MOT benchmark requires positive
        ret_item.append(d[0])
        ret_item.append(d[1])
        ret_item.append(trk.det_idx)
        ret_item.append(trk.confidence)
        ret.append(np.array(ret_item).reshape((1,-1)))

    #remove dead tracklets
    for t in range(len(self.trackers)-1,-1,-1):
      dead_flag_age = (self.trackers[t].age_since_update >= self.max_age_since_update)
      dead_flag_mov = (np.sqrt(((self.trackers[t].det_x-self.trackers[t].kf.x[:2])**2).sum()) > self.max_mov_since_update)
      if dead_flag_mov or dead_flag_age:
        self.trackers.pop(t)

    if(len(ret)>0):
      return np.concatenate(ret)
    return ret

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='SORT demo')
  parser.add_argument("--input",help="path to MOT dataset")
  parser.add_argument("--output",help="output path to save tracking results")
  parser.add_argument("--config",help="configuration JSON file")
  args = parser.parse_args()
  total_time = 0.0
  total_frames = 0

  with open(args.config) as fparam:
    param = json.loads(jsmin(fparam.read()))["sort"]

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  sequences = os.listdir(args.input)
  print("")
  for seq in sequences:
    if os.path.exists('%s/%s.txt'%(args.output,seq)): continue
    
    mot_tracker = Sort(
      max_age_since_update=param['max_age_after_last_update'],
      max_mov_since_update=param['max_mov_after_last_update'],
      d2t_dist_threshold_tight=param['d2t_dist_threshold_tight'],
      d2t_dist_threshold_loose=param['d2t_dist_threshold_loose'],
      motion_model_variance=param['motion_model_variance'],
      observation_variance=param['observation_variance'],
      motion_init_pose=param['motion_init_pose'],
      motion_init_sync=param['motion_init_sync'],
    ) #create instance of the SORT tracker

    if param['bypass_tracking_use_gt']:
      seq_points = np.loadtxt('%s/%s/gt/gt.txt'%(args.input,seq),delimiter=',') #load detections
    else:
      seq_points = np.loadtxt('%s/%s/det/det.txt'%(args.input,seq),delimiter=',') #load detections

    seq_points[:,7] *=1e-6 #convert micro-seconds to seconds

    print("Processing %s"%(seq))

    if param['bypass_tracking_use_gt']:
      print('\tWARNING: using ground-truth data (no tracking)')

    start_time = time.time()
    with open('%s/%s.txt'%(args.output,seq),'w') as out_file:
      for frame_idx,frame in enumerate(sorted(set(seq_points[:,0]))):
        points = seq_points[seq_points[:,0]==frame,:]
        if param['bypass_tracking_use_gt']:
          for d in points:
            print('%05d,%05d,%011.5f,%011.5f,%05d,1.00'%(frame,d[1],d[2],d[3],d[6]),file=out_file)
        else:
          tracks = mot_tracker.update(points[:,2:8]) #[x_loc, y_loc, x_vel, y_vel, det_idx, timestamp]
          for d in tracks:
            print('%05d,%05d,%011.5f,%011.5f,%05d,%.2f'%(frame,d[0],d[1],d[2],d[3],d[4]),file=out_file)

    total_time += time.time() - start_time
    total_frames += seq_points.shape[0]
  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))

