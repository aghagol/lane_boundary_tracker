import numpy as np
import os
import haversine

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
        #find detections that are too close to each other (to be removed)
        mark_for_deletion = []
        for i in range(points.shape[0]-1):
          for j in range(i+1,points.shape[0]):
            if haversine.dist(points[i,1],points[i,2],points[j,1],points[j,2])<parameters['min_det_dist']:
              if points[i,5]>=points[j,5]: #pick the point with higher confidence
                mark_for_deletion.append(j)
              else:
                mark_for_deletion.append(i)
        dets.append(np.delete(points,mark_for_deletion,axis=0))
  dets = np.vstack(dets)

  #sort detections according to timestamps
  dets = dets[dets[:,0].argsort(),:]

  #add detection index in a new column
  dets = np.hstack((np.arange(dets.shape[0]).reshape(-1,1)+1,dets))

  #remove duplicates (nearest neighbor?)

  #save result to CSV file
  fmt = ['%05d','%d','%.10f','%.10f','%.10f']
  with open(output_file_path,'w') as fout:
    np.savetxt(fout,dets[:,:5],fmt=fmt,delimiter=',')
