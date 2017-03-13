"""
This script creates standard MOT detections file
input: a CSV with columns: longitude, latitude, altitude, timestamp
output: a CSV with columns: frame#, -1, x, y, w, h, confidence, -1, -1, -1
"""
import numpy as np
import pandas as pd
import os

def make_MOT_det(input_file_path, output_file_path, parameters):
  """
  Input: a pose.csv file (chucai's format)
  Output: MOT-formatted det.txt

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

  # convert (longitude, latitude) to pixels
  lat = np.floor((lat-lat.min())/(lat.max()-lat.min())*parameters['image_nrows'])
  lon = np.floor((lon-lon.min())/(lon.max()-lon.min())*parameters['image_ncols'])

  # synthesize multiple tracks (by translating the pose track)
  
  tracks = []
  for shift in parameters['pose_shifts']:
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
  out[:,4] = parameters['object_size']
  out[:,5] = parameters['object_size']
  out[:,6] = 1
  out[:,7] = -1
  out[:,8] = -1
  out[:,9] = -1
  out = pd.DataFrame(out)
  out.to_csv(output_file_path, header=None, index=False)
 
if __name__ == "__main__":

  param = {}
  param['pose_step'] = 10
  param['image_nrows'] = 1000
  param['image_ncols'] = 1000
  param['object_size'] = 100
  param['pose_shifts'] = [(0,0),(-10,0),(10,0)]

  datadir = '/home/mo/Desktop/HADComplianceGroundTruth1/'
  print('Working on %s...'%datadir)
  for drive in os.listdir(datadir):
    if os.path.isdir(datadir+drive):
      print('Working on drive %s'%drive)
      counter = 0
      for input_file in os.listdir(datadir+drive):
        if input_file.endswith('pose.csv'):
          counter +=1
          if counter>1:
            print('Warning: found more than one pose csv in drive %s!'%drive)
            print('Processing the first one only...')
          else:
            out_path = 'out/%s/'%drive
            os.makedirs(out_path)
            make_MOT_det(datadir+drive+'/'+input_file, out_path+'det.txt', param)


