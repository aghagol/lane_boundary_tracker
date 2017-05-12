import numpy as np
import os
from haversine import dist

def index_TLLA_points(input_path,output_path,clusters,tiny_subdrives,parameters):
  """
  Input: Chen's CSV input format (a CSV file for each RANSAC output)
  Output: CSV file consisting of all detections
  """
  for subdrive in clusters:
    filelist = clusters[subdrive]

    dets = []
    for filename in filelist:
      if os.stat(input_path+filename).st_size:
        points = np.loadtxt(input_path+filename,delimiter=',').reshape(-1,5)
        points = points[points[:,4]%parameters['scanline_step']==0,:]
        dets.append(points)
    dets = np.vstack(dets)

    if dets.shape[0]<2:
      print('\tERROR: Marking %s for deletion due to insufficient points!'%(subdrive))
      tiny_subdrives.add(subdrive)
      continue

    #apply recall
    if parameters['recall']<1:
      dets = dets[np.random.rand(dets.shape[0])<parameters['recall'],:]

    #sort detections according to timestamps
    dets = dets[dets[:,0].argsort(),:]

    #find detections that are very close to each other (and mark for deletion)
    fake_confidence = np.random.rand(dets.shape[0])
    if parameters['remove_adjacent_points']:
      mark_for_deletion = []
      search_window_size = parameters['search_window_size']
      for i in range(dets.shape[0]):
        for j in range(max(i-search_window_size,0),min(i+search_window_size,dets.shape[0])):
          if dist(dets[i,1],dets[i,2],dets[j,1],dets[j,2])<parameters['min_pairwise_dist']:
            if fake_confidence[i]<fake_confidence[j]: #keep the point with higher confidence
              mark_for_deletion.append(i)
      dets = np.delete(dets,mark_for_deletion,axis=0)

    #add detection index in a new column
    dets = np.hstack((np.arange(dets.shape[0]).reshape(-1,1)+1,dets))

    #save result to CSV file
    os.makedirs(output_path+'%s/det/'%(subdrive))
    fmt = ['%05d','%d','%.10f','%.10f']
    with open(output_path+'%s/det/itll.txt'%(subdrive),'w') as fout:
      np.savetxt(fout,dets[:,:4],fmt=fmt,delimiter=',')
