import numpy as np
import pandas as pd

def store_json_timestamps(input_file_path, output_file_path):
  """
  Store a mapping between frame numbers and timestamps
  """
  import json
  with open(input_file_path) as data_file:
    timestamps = sorted([int(k) for k in json.load(data_file)['fromPosePointToSamplePoints']])
  timestamps = np.array([range(1,len(timestamps)+1),timestamps]).T
  pd.DataFrame(timestamps).to_csv(output_file_path, header=None, index=False)
