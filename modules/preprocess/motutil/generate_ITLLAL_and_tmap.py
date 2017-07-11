import numpy as np
import os
from haversine import dist

def generate_ITLLAL_and_tmap(input_path,output_path,clusters,tiny_subdrives,parameters):
  """
  Input: fuse files
  Output: itllal.txt, tmap.txt
  """
  for subdrive in clusters:

    #skip if output files exist for this subdrive
    if os.path.exists(output_path+'/%s/det/itllal.txt'%(subdrive)) and os.path.exists(output_path+'/%s/det/tmap.txt'%(subdrive)): continue

    #list of fuse files for this subdrive
    filelist = clusters[subdrive]

    #stack the points from all fuse files in filelist
    dets = []
    tmap = []
    for filename in filelist:
      if os.stat(input_path+'/'+filename).st_size:
        dets.append(np.loadtxt(input_path+'/'+filename,delimiter=',').reshape(-1,4))
        if parameters['fake_timestamp']:
          tmap.append(np.loadtxt(input_path+'/'+filename+'.tmap',delimiter=',').reshape(-1,2))
    dets = np.vstack(dets)
    if parameters['fake_timestamp']:
      tmap = np.vstack(tmap)

    #apply recall if recall < 100%
    if parameters['recall']<1:
      dets = dets[np.random.rand(dets.shape[0])<parameters['recall'],:]

    #re-arrange detections according to timestamps
    dets = dets[dets[:,0].argsort(),:]

    #find detections that are very close to each other (and mark for deletion)
    if parameters['remove_adjacent_points']:
      if True: #in case detection point confidences are unknown
        fake_confidence = np.random.rand(dets.shape[0])
      mark_for_deletion = []
      search_window_size = parameters['search_window_size']
      for i in range(dets.shape[0]):
        for j in range(max(i-search_window_size,0),min(i+search_window_size,dets.shape[0])):
          if dist(dets[i,2],dets[i,1],dets[j,2],dets[j,1])<parameters['min_pairwise_dist']:
            if fake_confidence[i]<fake_confidence[j]: #keep the point with higher confidence
              mark_for_deletion.append(i)
      dets = np.delete(dets,mark_for_deletion,axis=0)

    #remove subdrives/sequences with less than 2 points left after above processes
    if dets.shape[0]<2:
      tiny_subdrives.add(subdrive)
      continue

    #generate the itllal array
    #format: index,timestamp,lat,lon,altitude,label
    itllal = np.empty((dets.shape[0],6))
    itllal[:,0] = np.arange(dets.shape[0])+1
    itllal[:,1:5] = dets
    itllal[:,5] = -1 #no GT labels

    #write itllal array to file
    if not os.path.exists(output_path+'/%s/det/'%(subdrive)):
      os.makedirs(output_path+'/%s/det/'%(subdrive))
    fmt = ['%05d','%d','%.10f','%.10f','%.10f','%02d']
    np.savetxt(output_path+'/%s/det/itllal.txt'%(subdrive),itllal,fmt=fmt,delimiter=',')

    #write tmap array to file
    if parameters['fake_timestamp']:
      np.savetxt(output_path+'/%s/det/tmap.txt'%(subdrive),tmap,fmt=['%d','%d'],delimiter=',')