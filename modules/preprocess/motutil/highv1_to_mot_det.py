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
  zoom = 1 / parameters['pixel_size']

  prefix = '/FuseToTLLAIOutput/'
  chunk_list = [i for i in os.listdir(input_file_path+prefix) if i.isdigit()]
  for chunk in chunk_list:
    for filename in os.listdir(input_file_path+prefix+chunk):
      if not filename.endswith('tllai'): continue
      try:
        dets = np.vstack([dets,np.loadtxt(input_file_path+prefix+chunk+'/'+filename,delimiter=',')])
      except NameError:
        dets = np.loadtxt(input_file_path+prefix+chunk+'/'+filename,delimiter=',')
  dets = dets[:,[1,2,3,0]] #re-order columns to mimic pose data

  #extract vehicle pose information
  pose = np.loadtxt(input_file_path+'/'+pose_filename)
  timestamp_to_frame_idx = {k:v for v,k in enumerate(pose[:,3],start=1)}

  #round detections' timestamps to closest ones in the pose file (nearest neighbor)
  for i in range(dets.shape[0]):
    dets[i,3] = pose[np.argmin(abs(pose[:,3]-dets[i,3])),3]

  dets = np.vstack([dets,pose])
  dets_lon = dets[:,0]
  dets_lat = dets[:,1]
  dets_timestamps = dets[:,3]

  lat_min, lat_max = (dets_lat.min(), dets_lat.max())
  lon_min, lon_max = (dets_lon.min(), dets_lon.max())

  w = haversine(lon_min,lat_min,lon_max,lat_min)
  h = haversine(lon_min,lat_min,lon_min,lat_max)
  image_nrows = max(h*zoom,1)
  image_ncols = max(w*zoom,1)

  if lat_max==lat_min:
    dets_row = np.ones_like(dets_lat)
  else:
    dets_row = (dets_lat-lat_min)/(lat_max-lat_min)*image_nrows

  if lon_max==lon_min:
    dets_col = np.ones_like(dets_lon)
  else:
    dets_col = (dets_lon-lon_min)/(lon_max-lon_min)*image_ncols

  # write to output
  out = np.zeros((len(dets_timestamps),10),dtype=object)
  out[:,0] = [timestamp_to_frame_idx[t]+1 for t in dets_timestamps]
  out[:,1] = -1
  out[:,2] = dets_row
  out[:,3] = dets_col
  out[:,4] = parameters['object_size'] *zoom
  out[:,5] = parameters['object_size'] *zoom
  out[:,6] = 1
  out[:,7] = -1
  out[:,8] = -1
  out[:,9] = -1

  out = pd.DataFrame(out)
  out.to_csv(output_file_path, header=None, index=False)

  return 0
