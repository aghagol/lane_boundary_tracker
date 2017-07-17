import numpy as np
# import pandas as pd
import os,sys
import haversine

def get_tagged(points,pose,pose_tmap,scale_meta,parameters):
  """
  points  : points numpy array 1 (no   timestamp) format: id, latitude, longitude
  pose    : points numpy array 2 (with timestamp) format: latitude, longitude, altitude, timestamp
  tagged  : points numpy array 1 (with timestamp) format: id, latitude, longitude, altitude, timestamp

  Note: pose lat-lon has been converted to meters of distance from origin (bottom left corner)
  """
  tagged = np.zeros((points.shape[0],5))

  #IMPORTANT: store the points' lat-lon values before they are overwritten
  tagged[:,:3] = points[:,:3]

  #keep a mapping between point timestamps in the original time space and the "fake" time space
  tagged_tmap = np.zeros((points.shape[0],2))

  #meterize points (lat-lon to meters conversion)
  points[:,1] = (points[:,1]-scale_meta[0])*scale_meta[2]
  points[:,2] = (points[:,2]-scale_meta[1])*scale_meta[3]

  #we flag points for exclusion when they are farther than a certain distance from the pose path
  OK = np.ones((points.shape[0]),dtype=bool)

  for i in range(points.shape[0]):

    #calculate the l2 distance from the current detection point to all pose points!
    l2_squared_dist = ((points[i,1:3]-pose[:,:2])**2).sum(axis=1)

    #rank pose points based on their distances to the querry point (current detection point)
    pose_id = np.argsort(l2_squared_dist)
    #note: pose_id is an array of pose point indices sorted based on their rank

    #search for 2 high rank pose points that are adjacent in the pose sequence 

    r1 = 0  #rank of pose point number 1
    r2 = 1  #rank of pose point number 2
    #note: pose point number 1 is always closer to querry than pose point number 2

    while abs(pose_id[r2]-pose_id[r1])>1 and np.all(pose[pose_id[r2],:2]==pose[pose_id[r1],:2]):
      if (r2-r1)>3: #require consecutive pose points to be less than 3 ranks apart
        r1 +=1
        r2 = r1+1
      else:
        r2 +=1
    #explanation: below is the order in which pairs of pose points are searched
    #pose point number 1 with rank 0, pose point number 2 with rank 1
    #pose point number 1 with rank 0, pose point number 2 with rank 2
    #pose point number 1 with rank 0, pose point number 2 with rank 3
    #pose point number 1 with rank 1, pose point number 2 with rank 2
    #pose point number 1 with rank 1, pose point number 2 with rank 3
    #pose point number 1 with rank 1, pose point number 2 with rank 4
    #pose point number 1 with rank 2, pose point number 2 with rank 3
    #pose point number 1 with rank 2, pose point number 2 with rank 4
    #...

    #the np.all(pose[pose_id[r2],:2]==pose[pose_id[r1],:2]) statement above is due to repetitive pose points

    p0,p1 = pose_id[r1],pose_id[r2]

    #exclude detection point if it is farther than a certain distance from the pose path
    if min(l2_squared_dist[[p0,p1]])>parameters['tag_max_dist']**2:
      OK[i] = False

    #extract parameters for interpolating timestamps
    #vector1: vector from the pose point 1 to the detection point
    #vector2: vector from the pose point 1 to the detection point
    vector1 = points[i,1:3]-pose[p0,:2]
    vector2 = pose[p1,:2]-pose[p0,:2]

    #formula for calculating the timestamp: 
    # (1-lambda)*(pose point 1 timestamp) + lambda*(pose point 2 timestamp)
    #where lambda is the ratio of projection of vector1 onto vector2

    lam = vector1.dot(vector2) / vector2.dot(vector2)
    #note: numpy.dot for 1-D arrays performs the vector inner product
    #note: lambda is always smaller than 0.5*|vector1|

    #don't extrapolate if lambda<0
    lam = max(lam,0)
    
    if parameters['fake_timestamp']:
      tagged[i,4] = (1-lam)*pose_tmap[pose[p0,3]] + lam*pose_tmap[pose[p1,3]]
      tagged_tmap[i,1] = (1-lam)*pose[p0,3] + lam*pose[p1,3]
    else: #use true timestamps
      tagged[i,4] = (1-lam)*pose[p0,3] + lam*pose[p1,3]

    #if detection points don't have altitude information, use altitude from the matched pose points
    if parameters['tag_altitude']:
      tagged[i,3] = (1-lam)*pose[p0,2] + lam*pose[p1,2] -parameters['pose_altitude_offset']
    #note: parameters['pose_altitude_offset'] corresponds to the height of the vehicle

  tagged_tmap[:,0] = tagged[:,4]
  return (tagged[OK],tagged_tmap[OK])

def meterize(pose):
  """
  This module replaces the longitude latitude with meters distance from origin (bottom left corner)
  It also returns metadata (scaling parameters) for lat-long to meters data conversion
  """
  pose_meterized = pose.copy()

  lat_min,lat_max = pose[:,0].min(),pose[:,0].max()
  lon_min,lon_max = pose[:,1].min(),pose[:,1].max()

  h = haversine.dist(lon_min,lat_min,lon_min,lat_max)
  w = haversine.dist(lon_min,lat_min,lon_max,lat_min)

  lat_scale = h/(lat_max-lat_min) if h>0 else 1.
  lon_scale = w/(lon_max-lon_min) if w>0 else 1.

  pose_meterized[:,0] = (pose_meterized[:,0]-lat_min)*lat_scale
  pose_meterized[:,1] = (pose_meterized[:,1]-lon_min)*lon_scale

  return (pose_meterized,(lat_min,lon_min,lat_scale,lon_scale))