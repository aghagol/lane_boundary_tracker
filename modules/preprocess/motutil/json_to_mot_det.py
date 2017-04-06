import numpy as np
import pandas as pd
from haversine import haversine

def json_to_mot_det(input_file_path, output_file_path, parameters):
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

  import json
  with open(input_file_path) as data_file:
    data = json.load(data_file)['fromPosePointToSamplePoints']
  timestamps = sorted([int(k) for k in data])

  det_lon = np.array([float(p['longitude']) for t in timestamps for p in data['%d'%t]['samplepoints']])
  det_lat = np.array([float(p['latitude'])  for t in timestamps for p in data['%d'%t]['samplepoints']])
  det_timestamps = [t for t in timestamps for p in data['%d'%t]['samplepoints']]
  timestamp_to_frame_idx = dict(zip(timestamps,range(len(timestamps))))

  lat_min, lat_max = (det_lat.min(), det_lat.max())
  lon_min, lon_max = (det_lon.min(), det_lon.max())

  w = haversine(lon_min,lat_min,lon_max,lat_min)
  h = haversine(lon_min,lat_min,lon_min,lat_max)
  image_nrows = h *zoom
  image_ncols = w *zoom
  parameters['image_nrows'] = max(int(image_nrows),parameters['image_nrows'])
  parameters['image_ncols'] = max(int(image_ncols),parameters['image_ncols'])
  
  if max(image_nrows,image_ncols)>10000:
    print('\tWidth= %f (meters)'%(w))
    print('\tHeight= %f (meters)'%(h))
    print('\tZoom= %f'%(zoom))
    print('\tOutput image is %d x %d (pixels)'%(image_nrows,image_ncols))
    return 1
  
  det_row = (det_lat-lat_min) / (lat_max-lat_min) *image_nrows
  det_col = (det_lon-lon_min) / (lon_max-lon_min) *image_ncols

  # write to output
  out = np.zeros((len(det_timestamps),10),dtype=object)
  out[:,0] = [timestamp_to_frame_idx[t]+1 for t in det_timestamps]
  out[:,1] = -1
  out[:,2] = det_row
  out[:,3] = det_col
  out[:,4] = parameters['object_size'] *zoom
  out[:,5] = parameters['object_size'] *zoom
  out[:,6] = 1
  out[:,7] = -1
  out[:,8] = -1
  out[:,9] = -1

  #drop detections at drop_rate rate
  if parameters['drop_rate']<1:
    out = out[np.random.rand(out.shape[0])>parameters['drop_rate'],:]
  if out.shape[0]<parameters['min_dets']:
    return 2

  out = pd.DataFrame(out)
  out.to_csv(output_file_path, header=None, index=False)
  return 0
