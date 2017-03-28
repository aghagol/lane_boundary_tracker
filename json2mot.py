import os, json
import pandas as pd
import numpy as np
from PIL import Image
from MOT_util import haversine

json_toy = 'HT085_1465209329-posecentric.json'
output_file_path = 'det_mot.txt'

param = {
  'skip_npoints':50,
  'pixel_size':1., #in meters
  'object_size':10., #in meters
  }
zoom = 1/param['pixel_size']

with open(json_toy) as data_file:
  data = json.load(data_file)['fromPosePointToSamplePoints']

det_lon = np.array([float(p['longitude']) for k,v in data.iteritems() for p in v['samplepoints']])
det_lat = np.array([float(p['latitude'])  for k,v in data.iteritems() for p in v['samplepoints']])
det_timestamps = [k for k,v in data.iteritems() for p in v['samplepoints']]
timestamp_to_frame_idx = dict(zip(data.keys(),range(len(data))))

lat_min, lat_max = (det_lat.min(), det_lat.max())
lon_min, lon_max = (det_lon.min(), det_lon.max())

w = int(haversine(lon_min,lat_min,lon_max,lat_min))
h = int(haversine(lon_min,lat_min,lon_min,lat_max))

print('\tWidth= %d (meters)'%(w))
print('\tHeight= %d (meters)'%(h))

image_nrows = h *zoom
image_ncols = w *zoom
print('\tZoom= %f'%(zoom))
print('\tOutput image is %d x %d (pixels)'%(image_nrows,image_ncols))

# map (longitude, latitude) to (row, column)
det_row = np.floor( (det_lat-lat_min) / (lat_max-lat_min) *image_nrows ) +1
det_col = np.floor( (det_lon-lon_min) / (lon_max-lon_min) *image_ncols ) +1

# write to output
out = np.zeros((len(det_timestamps),10),dtype=int)
out[:,0] = [timestamp_to_frame_idx[t]+1 for t in det_timestamps]
out[:,1] = -1
out[:,2] = det_row
out[:,3] = det_col
out[:,4] = param['object_size'] *zoom
out[:,5] = param['object_size'] *zoom
out[:,6] = 1
out[:,7] = -1
out[:,8] = -1
out[:,9] = -1
out = pd.DataFrame(out)
out.to_csv(output_file_path, header=None, index=False)








