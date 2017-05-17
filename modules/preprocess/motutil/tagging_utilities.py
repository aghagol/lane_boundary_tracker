import numpy as np
# import pandas as pd
import os,sys
import haversine

def get_tagged(points,pose,scale_meta,tmap_pose,parameters):
  """
  points  : points numpy array 1 (without timestamp) format: longitude latitude laneline# scanline#
  pose    : points numpy array 2 (with timestamp)    format: longitude latitude altitude  timestamp
  tagged  : points numpy array 1 (with timestamp)    format: timestamp longitude latitude altitude laneline# scanline# [+gradient]
  """
  tagged = np.zeros((points.shape[0],6))
  tagged[:,1:3] = points[:,0:2]
  tagged[:,4:6] = points[:,2:4]

  tmap = np.zeros((points.shape[0],2))

  #normalize points as well
  points[:,0] = (points[:,0]-scale_meta[0])*scale_meta[2]
  points[:,1] = (points[:,1]-scale_meta[1])*scale_meta[3]

  OK = np.ones((points.shape[0]),dtype=bool)
  for i in range(points.shape[0]):

    l2_squared_dist = ((points[i,:2]-pose[:,:2])**2).sum(axis=1)
    pose_id = np.argsort(l2_squared_dist) #rank pose points based on distance to querry point

    #in addition to pose point with highest rank, find its adjacent pose point with highest rank
    r1 = 0  #rank of pose point number 1
    r2 = 1  #rank of pose point number 2
    #pose point number 1 is always closer to querry than pose point number 2
    while abs(pose_id[r2]-pose_id[r1])>1 and np.all(pose[pose_id[r2],:2]==pose[pose_id[r1],:2]):
      if (r2-r1)>3:
        r1 +=1
        r2 = r1+1
      else:
        r2 +=1

    p0,p1 = pose_id[r1],pose_id[r2]
    if l2_squared_dist[p1]>parameters['tag_max_dist'] or l2_squared_dist[p0]>parameters['tag_max_dist']:
      OK[i] = False

    #extract parameters for interpolating timestamps
    pose_points_dist = (pose[p1,:2]-pose[p0,:2]).dot(pose[p1,:2]-pose[p0,:2])
    lam = (points[i,:2]-pose[p0,:2]).dot(pose[p1,:2]-pose[p0,:2]) / pose_points_dist #0<lam<1
    lam = max(lam,0) #don't extrapolate
    
    if parameters['fake_timestamp']:
      tagged[i,0] = (1-lam)*tmap_pose[pose[p0,3]] + lam*tmap_pose[pose[p1,3]]
      tagged[i,3] = (1-lam)*pose[p0,2] + lam*pose[p1,2]
      tmap[i,0] = tagged[i,0]
      tmap[i,1] = lam*(pose[p1,3]-pose[p0,3])+pose[p0,3] #true timestamp
    else:
      tagged[i,0] = lam*(pose[p1,3]-pose[p0,3])+pose[p0,3]

  return (tagged[OK],tmap)

def meterize(pose):
  """
  This module replaces the longitude latitude with meters distance from origin
  and returns metadata to meterize any other lat-long data
  """
  lon_min,lon_max = pose[:,0].min(),pose[:,0].max()
  lat_min,lat_max = pose[:,1].min(),pose[:,1].max()

  w = haversine.dist(lon_min,lat_min,lon_max,lat_min)
  h = haversine.dist(lon_min,lat_min,lon_min,lat_max)

  lon_scale = w/(lon_max-lon_min) if w>0 else 1.
  lat_scale = h/(lat_max-lat_min) if h>0 else 1.

  pose[:,0] = (pose[:,0]-lon_min)*lon_scale
  pose[:,1] = (pose[:,1]-lat_min)*lat_scale

  return (lon_min,lat_min,lon_scale,lat_scale)