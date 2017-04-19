import numpy as np
import pandas as pd
import os
from haversine import haversine

def highv1_to_mot_det(input_file_path, pose_filename, output_file_path, parameters):
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

  #store all detections in a single numpy array
  prefix = '/FuseToTLLAIOutput/'
  chunk_list = [i for i in os.listdir(input_file_path+prefix) if i.isdigit()]
  for chunk in chunk_list:
    for filename in os.listdir(input_file_path+prefix+chunk):
      if not filename.endswith('tllai'): continue
      try:
        dets = np.vstack([dets,np.loadtxt(input_file_path+prefix+chunk+'/'+filename,delimiter=',')])
      except NameError:
        dets = np.loadtxt(input_file_path+prefix+chunk+'/'+filename,delimiter=',')
  dets = dets[:,[1,2,3,0]] #re-order columns according to pose format

  #extract vehicle pose information
  pose = np.loadtxt(input_file_path+'/'+pose_filename)
  pose = pose[pose[:,3].argsort(),:] #sort points based on timestamps (probably already sorted)
  timestamp_id = {k:v for v,k in enumerate(pose[:,3])}

  #smooth the pose
  if parameters['smooth_pose']:
    from scipy.ndimage.filters import gaussian_filter1d
    pose[:,0:2] = gaussian_filter1d(pose[:,0:2],sigma=10,axis=0,mode='nearest')

  #round detections' timestamps to closest ones in the pose file (nearest neighbor)
  for i in range(dets.shape[0]):
    dets[i,3] = pose[np.argmin(abs(pose[:,3]-dets[i,3])),3]

  #sort detections according to timestamps (probably already sorted)
  dets = dets[dets[:,3].argsort(),:]

  #augment detections with fake (pose-based) points
  if parameters['fake_dets']:
    dets_aug = []
    for i in range(dets.shape[0]):
      dets_aug.append(dets[i,:]) #add the detection itself
      det_timestamp_id = timestamp_id[dets[i,3]]
      lla_offset = dets[i,:]-pose[det_timestamp_id,:]
      discard_fake_point = False
      for j in range(parameters['fake_dets_n']):
        fake_det = pose[det_timestamp_id+(j*parameters['pose_step']),:]+lla_offset
        for k in range(i+1,min(i+100,dets.shape[0])): #check if future detections are close by
          if haversine(dets[k,0],dets[k,1],fake_det[0],fake_det[1])<parameters['fake_dets_min_dist']:
            discard_fake_point = True
            break
        if discard_fake_point: break
        dets_aug.append(fake_det)
    dets = np.vstack(dets_aug)
    dets = dets[dets[:,3].argsort(),:]

  #merge detections with pose (if desired) - for KF state initialization
  if parameters['start_with_pose']:
    start_idx = max(timestamp_id[dets[0,3]]-10,0)
    stop_idx = min(start_idx+10,pose.shape[0])
    dets = np.vstack([dets,pose[start_idx:stop_idx:parameters['pose_step']]])
    dets = dets[dets[:,3].argsort(),:]

  lon_min, lon_max = (dets[:,0].min(), dets[:,0].max())
  lat_min, lat_max = (dets[:,1].min(), dets[:,1].max())
  w = haversine(lon_min,lat_min,lon_max,lat_min)
  h = haversine(lon_min,lat_min,lon_min,lat_max)

  # write to output
  out = np.zeros((dets.shape[0],10),dtype=object)
  out[:,0] = [timestamp_id[t]-timestamp_id[dets[0,3]]+1 for t in dets[:,3]]
  out[:,1] = -1
  out[:,2] = (dets[:,1]-lat_min)/(lat_max-lat_min)*h*zoom
  out[:,3] = (dets[:,0]-lon_min)/(lon_max-lon_min)*w*zoom
  out[:,4] = parameters['object_size'] *zoom
  out[:,5] = parameters['object_size'] *zoom
  out[:,6] = 1
  out[:,7] = -1
  out[:,8] = -1
  out[:,9] = -1

  out = pd.DataFrame(out)
  out.to_csv(output_file_path, header=None, index=False)

  return 0
