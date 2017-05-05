import numpy as np
import os
from haversine import dist

def index_TLLA_points(input_file_path,output_file_path,chunk_times,parameters):
  """
  Input: Chen's CSV input format (a CSV file for each RANSAC output)
  Output: CSV file consisting of all detections
  """
  #store all detections in a single numpy array
  prefix = '/FuseToTLLAIOutput/'
  chunk_list = sorted([int(i) for i in os.listdir(input_file_path+prefix) if i.isdigit()])
  dets = []
  for chunk_idx,chunk in enumerate(chunk_list):
    chunk_path = input_file_path+prefix+'%d'%chunk
    for filename in os.listdir(chunk_path):
      if filename.endswith('tllai'):
        points = np.loadtxt(chunk_path+'/'+filename,delimiter=',').reshape(-1,6)
        points = points[points[:,0].argsort(),:]
        if parameters['reduce_short_ransac_lines']:
          if points.shape[0]>1:
            if dist(points[0,1],points[0,2],points[-1,1],points[-1,2])<parameters['min_line_length']:
              points = points[0,:].reshape(1,6)
        if parameters['remove_chunk_overflow']:
          t0 = int(chunk_times[chunk_times['ChunkId']==chunk]['StartTime'])
          t1 = int(chunk_times[chunk_times['ChunkId']==chunk]['EndTime'])
          points = points[np.logical_and(points[:,0]>=t0,points[:,0]<t1),:]
        if parameters['rank_reduction']:
          if points.shape[0]>2:
            U,S,V = np.linalg.svd(points[:,1:3]-points[:,1:3].mean(axis=0),full_matrices=False)
            points[:,1:3] = points[:,1:3].mean(axis=0) + S[0]*U[:,:1].dot(V[:1,:])
        dets.append(points)
  dets = np.vstack(dets)

  #sort detections according to timestamps
  dets = dets[dets[:,0].argsort(),:]

  #find detections that are very close to each other (and mark for deletion)
  if parameters['remove_adjacent_endpoints']:
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
