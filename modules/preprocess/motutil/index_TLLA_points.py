import numpy as np
import os
from haversine import dist

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
        points = np.loadtxt(input_file_path+prefix+chunk+'/'+filename,delimiter=',').reshape(-1,6)
        if parameters['rank_reduction']:
          if points.shape[0]>1:
            pass
        dets.append(points)
  dets = np.vstack(dets)

  #sort detections according to timestamps
  dets = dets[dets[:,0].argsort(),:]

  #find detections that are very close to each other (and mark for deletion)
  if parameters['remove_adjacent_points']:
    mark_for_deletion = []
    search_window_size = parameters['search_window_size']
    for i in range(dets.shape[0]):
      for j in range(max(i-search_window_size,0),min(i+search_window_size,dets.shape[0])):
        if dist(dets[i,1],dets[i,2],dets[j,1],dets[j,2])<parameters['min_pairwise_dist']:
          if dets[i,5]<dets[j,5]: #keep the point with higher confidence
            mark_for_deletion.append(i)
    dets = np.delete(dets,mark_for_deletion,axis=0)

  #add detection index in a new column
  dets = np.hstack((np.arange(dets.shape[0]).reshape(-1,1)+1,dets))

  #save result to CSV file
  fmt = ['%05d','%d','%.10f','%.10f','%.10f']
  with open(output_file_path,'w') as fout:
    np.savetxt(fout,dets[:,:5],fmt=fmt,delimiter=',')
