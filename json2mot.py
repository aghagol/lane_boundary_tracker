import os, json
import pandas as pd
import numpy as np
from PIL import Image

json_toy = 'HT085_1465209329-posecentric.json'
param = {}
param['pose_step'] = 50
param['image_nrows_default'] = 1000 #None for automatic
param['image_ncols_default'] = 1000 #None for automatic
param['object_size'] = 100 #in auto mode, this is in meters

with open(json_toy) as data_file:    
    data = json.load(data_file)['fromPosePointToSamplePoints']

pose_timestamps = data.keys()

for t in pose_timestamps:
    intersection_points = data[t]['samplepoints']
    for ip in intersection_points:
        ip_lon = float(ip['longitude'])
        ip_lat = float(ip['latitude'])
        ip_alt = float(ip['elevation'])
        ip_surface_id = long(ip['roadSurfaceID'])
        ip_LB_id = ip['boundaryCurveID']


# from old script...

# pose_df = pd.read_csv(input_file_path, header=None, sep=' ')
# pose_df = pose_df.ix[::parameters['pose_step']]

# lat = np.array(pose_df[0]) #latitude
# lon = np.array(pose_df[1]) #longitude
n_frames = len(pose_timestamps)

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








