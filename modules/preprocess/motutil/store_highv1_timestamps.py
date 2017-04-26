import numpy as np
import pandas as pd

def store_highv1_timestamps(pose_path,output_file_path,parameters):
  """
  Store a mapping between frame numbers and timestamps
  """
  timestamps = np.loadtxt(pose_path)[:,3] #LLAT format
  n = timestamps.shape[0]
  timestamps = timestamps.reshape(n,1)
  timestamps = np.hstack([np.arange(n).reshape(-1,1)+1,timestamps])
  fmt = ['%05d','%d']
  with open(output_file_path,'w') as fout:
  	np.savetxt(fout,timestamps,fmt=fmt,delimiter=',')
