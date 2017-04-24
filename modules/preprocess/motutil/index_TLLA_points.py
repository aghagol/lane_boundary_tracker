import numpy as np
import pandas as pd
import os
from haversine import haversine

def index_TLLA_points(input_file_path,output_file_path,parameters):
  """
  Input: Chen's CSV input format (a CSV file for each RANSAC output)
  Output: CSV file consisting of all detections
  """
  #store all detections in a single numpy array
  prefix = '/FuseToTLLAIOutput/'
  chunk_list = [i for i in os.listdir(input_file_path+prefix) if i.isdigit()]
  dets = []
  for chunk in chunk_list:
    for filename in os.listdir(input_file_path+prefix+chunk):
      if filename.endswith('tllai'):
        dets.append(np.loadtxt(input_file_path+prefix+chunk+'/'+filename,delimiter=','))
  dets = np.vstack(dets)

  #sort detections according to timestamps
  dets = dets[dets[:,0].argsort(),:]

  #add detection index in a new column
  dets = np.hstack((np.arange(dets.shape[0]).reshape(-1,1)+1,dets))

  #remove duplicates

  #save result to CSV file
  fmt = ['%05d','%d','%.10f','%.10f','%.10f']
  with open(output_file_path,'w') as fout:
    np.savetxt(fout,dets[:,:5],fmt=fmt,delimiter=',')
