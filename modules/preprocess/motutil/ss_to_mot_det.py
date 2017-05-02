import numpy as np
# import pandas as pd
# import os
import haversine

def ss_to_mot_det(output_path,drive,parameters):
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

  #load detections from txt file
  dets = np.loadtxt(output_path+'%s/det/tlla.txt'%(drive),delimiter=',')

  #extract vehicle pose information
  if parameters['video_tracking']:
    pose = np.loadtxt(pose_path)[:,[3,0,1]] #save in TLL format
    pose = np.hstack((np.zeros((pose.shape[0],1)).reshape(-1,1),pose)) #add a column for flags
    pose = pose[pose[:,1].argsort(),:] #sort points based on timestamps (probably already sorted)
    timestamp_id = {k:v for v,k in enumerate(pose[:,1])}
    for i in range(dets.shape[0]): #round detections' timestamps to closest ones in the pose file
      dets[i,1] = pose[np.argmin(abs(pose[:,1]-dets[i,1])),1]
    dets = dets[dets[:,1].argsort(),:]
    #smooth the pose
    if parameters['smooth_pose']:
      from scipy.ndimage.filters import gaussian_filter1d
      pose[:,2:4] = gaussian_filter1d(pose[:,2:4],sigma=10,axis=0,mode='nearest')    
    #augment detections with fake (pose-based) points
    if parameters['fake_dets']:
      dets_aug = []
      for i in range(dets.shape[0]):
        dets_aug.append(dets[i,:].reshape(1,-1)) #add the detection itself
        det_timestamp_id = timestamp_id[dets[i,1]]
        lla_offset = dets[i,:]-pose[det_timestamp_id,:]
        lla_offset[0] = 0 #since this is a guide (fake detection)
        discard_fake_point = False
        for j in range(1,parameters['fake_dets_n']+1):
          fake_det = pose[det_timestamp_id+(j*parameters['fake_dets_pose_step']),:]+lla_offset
          for k in range(i+1,min(i+100,dets.shape[0])): #check if future detections are close by
            if haversine.dist(dets[k,2],dets[k,3],fake_det[2],fake_det[3])<parameters['fake_dets_min_dist']:
              discard_fake_point = True
              break
          if discard_fake_point: break
          dets_aug.append(fake_det.reshape(1,-1))
      dets = np.vstack(dets_aug)
      dets = dets[dets[:,1].argsort(),:] #re-sort!
    #add pose points for KF initialization
    if parameters['start_with_pose']:
      start_idx = max(timestamp_id[dets[0,1]]-parameters['head_start'],0)
      stop_idx = min(start_idx+1,pose.shape[0]-1)
      pose_guide = pose[start_idx:stop_idx:parameters['pose_step'],:]
      dets = np.vstack([pose_guide,dets])
      dets = dets[dets[:,1].argsort(),:] #re-sort
    video_frame_ids = [timestamp_id[t]-timestamp_id[dets[0,1]]+1 for t in dets[:,1]]

  lon_min, lon_max = (dets[:,2].min(), dets[:,2].max())
  lat_min, lat_max = (dets[:,3].min(), dets[:,3].max())
  w = haversine.dist(lon_min,lat_min,lon_max,lat_min)
  h = haversine.dist(lon_min,lat_min,lon_min,lat_max)

  # write to output
  fmt = ['%05d','%d','%011.5f','%011.5f','%05d','%05d','%05d','%d']
  out = np.zeros((dets.shape[0],8))
  out[:,0] = range(1,out.shape[0]+1)
  out[:,1] = -1
  out[:,2] = (dets[:,3]-lat_min)/(lat_max-lat_min)*h*zoom
  out[:,3] = (dets[:,2]-lon_min)/(lon_max-lon_min)*w*zoom
  out[:,4] = parameters['object_size'] *zoom
  out[:,5] = parameters['object_size'] *zoom
  out[:,6] = dets[:,0] #detection point's uniquely associated indices
  out[:,7] = dets[:,1] #detection point's timestamp
  if parameters['video_tracking']: out[:,0] = video_frame_ids
  
  with open(output_path+'%s/det/det.txt'%(drive),'w') as fout:
    np.savetxt(fout,out,fmt=fmt,delimiter=',')
