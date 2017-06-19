import numpy as np
# import pandas as pd
import os,sys
import haversine

def ss_to_mot_gt(output_path,clusters,tiny_subdrives,pose_path,parameters):
  """
  Input: a JSON file (chucai's format)
  Output: MOT-formatted det.txt

  usage:
    JSON_to_MOT_det(input_file_path, output_file_path, parameters)

  parameters must be a dictionary with following keys:
    pixel_size................pixel size in meters
    object_size...............target size in meters
  """
  zoom = 1. / parameters['pixel_size']

  if parameters['motion_observations']:
    pose = np.loadtxt(pose_path)[:,[3,0,1,2]] #in TLL format
    pose = pose[pose[:,0].argsort(),:] #sort based on timestamp
    if parameters['fake_timestamp']:
      pose[:,0] = np.arange(pose.shape[0])*1e6 #constant speed model

  for subdrive in clusters:
    if os.path.exists(output_path+'%s/gt/gt.txt'%(subdrive)): continue
    if subdrive in tiny_subdrives: continue

    #load detections from txt file
    dets = np.loadtxt(output_path+'%s/det/itllal.txt'%(subdrive),delimiter=',')

    lat_min, lat_max = (dets[:,2].min(), dets[:,2].max())
    lon_min, lon_max = (dets[:,3].min(), dets[:,3].max())
    h = haversine.dist(lon_min,lat_min,lon_min,lat_max)
    w = haversine.dist(lon_min,lat_min,lon_max,lat_min)

    lat_scale = h*zoom/(lat_max-lat_min) if h>0 else 1.
    lon_scale = w*zoom/(lon_max-lon_min) if w>0 else 1.

    #replace bounding-boxes with motion observations from pose
    if parameters['motion_observations']:
      motion = np.zeros((dets.shape[0],2))
      for i in range(dets.shape[0]):
        j = np.argmax(dets[i,1]<pose[:,0]) #return index of matched pose point
        if j>0:
          motion[i,0] = (pose[j,1]-pose[j-1,1])*lat_scale/(pose[j,0]-pose[j-1,0])*1e6
          motion[i,1] = (pose[j,2]-pose[j-1,2])*lon_scale/(pose[j,0]-pose[j-1,0])*1e6
        else:
          motion[i,:] = 0

    # write to output
    fmt = ['%05d','%d','%011.5f','%011.5f','%07.5f','%07.5f','%05d','%d']
    out = np.zeros((dets.shape[0],8))
    out[:,0] = range(1,out.shape[0]+1) #frame number
    out[:,1] = dets[:,5]
    out[:,2] = (dets[:,2]-lat_min)*lat_scale #row (sub-pixel)
    out[:,3] = (dets[:,3]-lon_min)*lon_scale #column (sub-pixel)
    out[:,6] = dets[:,0] #detection's unique ID
    out[:,7] = dets[:,1] #detection's timestamp
    if parameters['motion_observations']:
      out[:,4:6] = motion
    else:
      out[:,4:6] = parameters['object_size'] *zoom

    if not os.path.exists(output_path+'%s/gt/'%(subdrive)):
      os.makedirs(output_path+'%s/gt/'%(subdrive))
    np.savetxt(output_path+'%s/gt/gt.txt'%(subdrive),out,fmt=fmt,delimiter=',')
