"""
This script creates standard MOT detections file
input: a CSV with columns: longitude, latitude, altitude, timestamp
output: a CSV with columns: frame#, -1, x, y, w, h, confidence, -1, -1, -1
"""
import numpy as np
import pandas as pd

# read data
posefile = '/home/mo/Desktop/HADComplianceGroundTruth1/HT010_1465209606/80416021-pose.csv'
pose_df = pd.read_csv(posefile,header=None,sep=' ')

# subsample
n_steps = 200
pose_df = pose_df.ix[::pose_df.shape[0]/n_steps]

# convert lla to pixels
image_nrows = 1000
image_ncols = 1000
object_size = 100

lat = np.array(pose_df[0])
lon = np.array(pose_df[1])
n_frames = lat.shape[0]

# normalize
lat = np.floor((lat-lat.min())/(lat.max()-lat.min())*image_nrows)
lon = np.floor((lon-lon.min())/(lon.max()-lon.min())*image_ncols)

# synthesize multiple tracks (by translating the pose track)
shifts = [(0,0),(-100,0),(100,0)]
tracks = []
for shift in shifts:
  tracks.append((lat+shift[0],lon+shift[1]))
n_tracks = len(tracks)

# write to output
out = np.zeros((n_frames*n_tracks,10),dtype=int)
for frame in range(n_frames):
  for ii in range(n_tracks):
    out[frame*n_tracks + ii,0] = frame
    out[frame*n_tracks + ii,2] = tracks[ii][0][frame]
    out[frame*n_tracks + ii,3] = tracks[ii][1][frame]
out[:,1] = -1
out[:,4] = object_size
out[:,5] = object_size
out[:,6] = 1
out[:,7] = -1
out[:,8] = -1
out[:,9] = -1
out = pd.DataFrame(out)
out.to_csv('../sort/data/pose/det.txt',header=None,index=False)