import numpy as np
# import pandas as pd
import os,sys
import haversine

def ss_to_mot_det(output_path,clusters,pose_path,parameters):
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
    pose = pose[pose[:,0].argsort(),:]

  for subdrive in clusters:

    #load detections from txt file
    dets = np.loadtxt(output_path+'%s/det/itll.txt'%(subdrive),delimiter=',')

    lon_min, lon_max = (dets[:,2].min(), dets[:,2].max())
    lat_min, lat_max = (dets[:,3].min(), dets[:,3].max())
    w = haversine.dist(lon_min,lat_min,lon_max,lat_min)
    h = haversine.dist(lon_min,lat_min,lon_min,lat_max)

    #replace bounding-boxes with motion observations from pose
    if parameters['motion_observations']:
      motion = np.zeros((dets.shape[0],2))
      for i in range(dets.shape[0]):
        j = np.argmax(dets[i,1]<pose[:,0]) #return index of matched pose point
        if j>0 and pose[j,0]>pose[j-1,0]:
          motion[i,1] = (pose[j,1]-pose[j-1,1])/(lon_max-lon_min)*w*zoom/(pose[j,0]-pose[j-1,0])
          motion[i,0] = (pose[j,2]-pose[j-1,2])/(lat_max-lat_min)*h*zoom/(pose[j,0]-pose[j-1,0])
        else:
          motion[i,:] = 0

    # write to output
    fmt = ['%05d','%d','%011.5f','%011.5f','%07.5f','%07.5f','%05d','%d']
    out = np.zeros((dets.shape[0],8))
    out[:,0] = range(1,out.shape[0]+1) #frame number
    out[:,1] = -1
    out[:,2] = (dets[:,3]-lat_min)/(lat_max-lat_min)*h*zoom #row
    out[:,3] = (dets[:,2]-lon_min)/(lon_max-lon_min)*w*zoom #column
    out[:,6] = dets[:,0] #detection's unique ID
    out[:,7] = dets[:,1] #detection's timestamp
    if parameters['motion_observations']:
      out[:,4:6] = motion
    else:
      out[:,4:6] = parameters['object_size'] *zoom
    
    with open(output_path+'%s/det/det.txt'%(subdrive),'w') as fout:
      np.savetxt(fout,out,fmt=fmt,delimiter=',')
