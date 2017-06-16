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
    tmap = []
    lb_id_offset = 0
    for filename in filelist:
      if os.stat(input_path+filename).st_size:
        points = np.loadtxt(input_path+filename,delimiter=',').reshape(-1,6)
        points = points[points[:,5]>parameters['confidence_thresh'],:]
        points[:,4] += lb_id_offset #re-assign IDs to avoid duplicate IDs for LBs from two images
        lb_id_offset = points[:,4].max()+1
        dets.append(points)
        tmap.append(np.loadtxt(input_path+filename+'.tmap',delimiter=',').reshape(-1,2))
    dets = np.vstack(dets)
    if parameters['fake_timestamp']: tmap = np.vstack(tmap)

    #apply recall
    if parameters['recall']<1:
      dets = dets[np.random.rand(dets.shape[0])<parameters['recall'],:]

    #sort detections according to timestamps
    dets = dets[dets[:,0].argsort(),:]

    #find detections that are very close to each other (and mark for deletion)
    if parameters['remove_adjacent_points']:
      fake_confidence = np.random.rand(dets.shape[0])
      mark_for_deletion = []
      search_window_size = parameters['search_window_size']
      for i in range(dets.shape[0]):
        for j in range(max(i-search_window_size,0),min(i+search_window_size,dets.shape[0])):
          if dist(dets[i,2],dets[i,1],dets[j,2],dets[j,1])<parameters['min_pairwise_dist']:
            if fake_confidence[i]<fake_confidence[j]: #keep the point with higher confidence
              mark_for_deletion.append(i)
      dets = np.delete(dets,mark_for_deletion,axis=0)

    if dets.shape[0]<2:
      print('\tERROR: Marking %s for deletion due to insufficient points!'%(subdrive))
      tiny_subdrives.add(subdrive)
      continue

    #add detection index in a new column
    itlla = np.hstack((np.arange(dets.shape[0]).reshape(-1,1)+1,dets[:,:5]))

    #save result to CSV file
    os.makedirs(output_path+'%s/det/'%(subdrive))
    fmt = ['%05d','%d','%.10f','%.10f','%.10f','%02d']
    np.savetxt(output_path+'%s/det/itlla.txt'%(subdrive),itlla,fmt=fmt,delimiter=',')
    if parameters['fake_timestamp']:
      np.savetxt(output_path+'%s/det/tmap.txt'%(subdrive),tmap,fmt=['%d','%d'],delimiter=',')
