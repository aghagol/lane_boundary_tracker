import numpy as np
import pandas as pd

def store_highv1_timestamps(input_file_path, pose_filename, output_file_path):
  """
  Store a mapping between frame numbers and timestamps
  """
  timestamps = np.loadtxt(input_file_path+'/'+pose_filename)[:,-1]
  timestamps = np.hstack([np.arange(timestamps.shape[0]).reshape(timestamps.shape[0],1)+1,timestamps])
  pd.DataFrame(timestamps).to_csv(output_file_path, header=None, index=False)
