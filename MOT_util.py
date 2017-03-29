import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    m = 1000 * 6367 * c
    return m

def make_MOT_det(input_file_path, output_file_path, parameters):
  """
  Input: a pose.csv file (chucai's format)
  Output: MOT-formatted det.txt

  Usage:
    make_MOT_det(input_file_path, output_file_path, parameters)

  parameters must be a dictionary with following keys:
    pose_step: param for subsampling pose sequence (frame rate)
    image_nrows: param for converting latitude to pixels
    image_ncols: param for converting longitude to pixels
    object_size: target size in pixels
    pose_shifts: param for pose cloning
  """
  pose_df = pd.read_csv(input_file_path, header=None, sep=' ')
  pose_df = pose_df.ix[::parameters['pose_step']]

  lat = np.array(pose_df[0]) #latitude
  lon = np.array(pose_df[1]) #longitude
  n_frames = lat.shape[0]

  if not parameters['image_nrows_default']:
    w = int(haversine(lon.min(),lat.min(),lon.max(),lat.min()))
    h = int(haversine(lon.min(),lat.min(),lon.min(),lat.max()))
    print('\tAutomatic scaling: width=%d (m), height=%d (m)'%(w,h))
    parameters['image_nrows'] = h+parameters['object_size'] # meter-wide pixels
    parameters['image_ncols'] = w+parameters['object_size']
    if max(w,h)>10000:
      return 1
  else:
    parameters['image_nrows'] = parameters['image_nrows_default']
    parameters['image_ncols'] = parameters['image_ncols_default']

  # convert (longitude, latitude) to pixels
  lat = np.floor((lat-lat.min())/(lat.max()-lat.min())*parameters['image_nrows'])+1
  lon = np.floor((lon-lon.min())/(lon.max()-lon.min())*parameters['image_ncols'])+1

  # synthesize multiple tracks (by translating the pose track)
  
  tracks = []
  for shift in parameters['pose_shifts']:
    tracks.append((lat+shift[0],lon+shift[1]))
  n_tracks = len(tracks)

  # write to output
  out = np.zeros((n_frames*n_tracks,10),dtype=int)
  for frame in range(n_frames):
    for ii in range(n_tracks):
      out[frame*n_tracks + ii,0] = frame+1
      out[frame*n_tracks + ii,2] = tracks[ii][0][frame]#-parameters['object_size']/2
      out[frame*n_tracks + ii,3] = tracks[ii][1][frame]#-parameters['object_size']/2
  out[:,1] = -1
  out[:,4] = parameters['object_size']
  out[:,5] = parameters['object_size']
  out[:,6] = 1
  out[:,7] = -1
  out[:,8] = -1
  out[:,9] = -1
  out = pd.DataFrame(out)
  out.to_csv(output_file_path, header=None, index=False)
  return 0

def JSON_to_MOT_det(input_file_path, output_file_path, parameters):
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

  print('\tWidth= %f (meters)'%(w))
  print('\tHeight= %f (meters)'%(h))

  image_nrows = h *zoom
  image_ncols = w *zoom
  parameters['image_nrows'] = int(image_nrows)
  parameters['image_ncols'] = int(image_ncols)
  print('\tZoom= %f'%(zoom))
  print('\tOutput image is %d x %d (pixels)'%(image_nrows,image_ncols))
  # if max(image_nrows,image_ncols)>10000:
  #   return 1

  # map (longitude, latitude) to (row, column)
  # det_row = np.floor( (det_lat-lat_min) / (lat_max-lat_min) *image_nrows ) +1
  # det_col = np.floor( (det_lon-lon_min) / (lon_max-lon_min) *image_ncols ) +1

  det_row = (det_lat-lat_min) / (lat_max-lat_min) *image_nrows
  det_col = (det_lon-lon_min) / (lon_max-lon_min) *image_ncols

  # write to output
  out = np.zeros((len(det_timestamps),10))
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
  out = pd.DataFrame(out)
  out.to_csv(output_file_path, header=None, index=False)
  return 0