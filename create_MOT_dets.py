"""
This script creates standard MOT detections file
input: a CSV with columns: longitude, latitude, altitude, timestamp
output: a CSV with columns: frame#, -1, x, y, w, h, confidence, -1, -1, -1
"""
import numpy as np
import pandas as pd

# posefile = '/home/mo/Desktop/HADComplianceGroundTruth1/HT010_1465209606/80416021-pose.csv'
posefile = '80416021-pose.csv'
pose_df = pd.read_csv(posefile,header=None,sep=' ')

# limit to a 1000 frames
step = int(pose_df.shape[0]/1000)
pose_df = pose_df.ix[::step]
pose_df = pose_df.ix[:1000*step]

# convert longitude, latitude to pixels
image_nrows = 1000
image_ncols = 1000

lat = np.array(pose_df[0])
lon = np.array(pose_df[1])
nframes = lat.shape[0]

# normalize
lat = np.floor((lat-lat.min())/(lat.max()-lat.min())*image_nrows)
lon = np.floor((lon-lon.min())/(lon.max()-lon.min())*image_ncols)

# create multiple tracks by translating the pose track
shifts = [(0,0),(-100,0),(100,0)]
tracks = []
for shift in shifts:
  tracks.append((lat+shift[0],lon+shift[1]))
ntracks = len(tracks)

# write to output
out = np.zeros((nframes*ntracks,10),dtype=int)
for frame in range(nframes):
  for ii in range(ntracks):
    out[frame*ntracks + ii,0] = frame
    out[frame*ntracks + ii,2] = tracks[ii][0][frame]
    out[frame*ntracks + ii,3] = tracks[ii][1][frame]
out[:,1] = -1
out[:,4] = 50
out[:,5] = 50
out[:,6] = 1
out[:,7] = -1
out[:,8] = -1
out[:,9] = -1

out = pd.DataFrame(out)

out.to_csv('../sort/data/pose/det.txt',header=None,index=False)