import numpy as np
import os,sys
import haversine

def generate_MOT_gt(output_path,clusters,tiny_subdrives,pose_path,parameters):
  """
  Input: fuse files, itllal.txt
  Output: MOT-formatted gt.txt
  """
  #for each detection point, extract the direction of travel from pose
  if parameters['motion_observations']:
    pose = np.loadtxt(pose_path)[:,[3,0,1,2]] #format: timestamp, latitude, longitude, altitude
    pose = pose[pose[:,0].argsort(),:] #re-arrange based on timestamp
    if parameters['fake_timestamp']:
      #consecutive pose points must be 1 second (1e6 microseconds) apart
      pose[:,0] = np.arange(pose.shape[0])*1e6

  for subdrive in clusters:

    #skip subdrive sequence if det.txt already exists
    if os.path.exists(output_path+'/%s/det/det.txt'%(subdrive)): continue

    #skip subdrive sequence if subdrive is marked as "tiny"
    if subdrive in tiny_subdrives: continue

    #read detection points from itllal.txt
    dets = np.loadtxt(output_path+'/%s/det/itllal.txt'%(subdrive),delimiter=',')

    lat_min, lat_max = (dets[:,2].min(), dets[:,2].max())
    lon_min, lon_max = (dets[:,3].min(), dets[:,3].max())
    h = haversine.dist(lon_min,lat_min,lon_min,lat_max)
    w = haversine.dist(lon_min,lat_min,lon_max,lat_min)

    lat_scale = h/(lat_max-lat_min) if h>0 else 1.
    lon_scale = w/(lon_max-lon_min) if w>0 else 1.

    #compute motion observations from pose for each detection point
    if parameters['motion_observations']:
      motion = np.zeros((dets.shape[0],2))
      for i in range(dets.shape[0]):
        j = np.argmax(dets[i,1]<pose[:,0]) #return index of matched pose point
        if j>0:
          motion[i,0] = (pose[j,1]-pose[j-1,1])*lat_scale/(pose[j,0]-pose[j-1,0])*1e6
          motion[i,1] = (pose[j,2]-pose[j-1,2])*lon_scale/(pose[j,0]-pose[j-1,0])*1e6
        else:
          motion[i,:] = 0

    # write to output
    fmt = ['%05d','%d','%011.5f','%011.5f','%07.5f','%07.5f','%05d','%d']
    out = np.zeros((dets.shape[0],8))
    out[:,0] = range(1,out.shape[0]+1) #frame number
    out[:,1] = dets[:,5] #this is set to -1 if GT labels do not exist
    out[:,2] = (dets[:,2]-lat_min)*lat_scale #row (sub-pixel)
    out[:,3] = (dets[:,3]-lon_min)*lon_scale #column (sub-pixel)
    out[:,6] = dets[:,0] #detection's unique ID
    out[:,7] = dets[:,1] #detection's timestamp
    if parameters['motion_observations']:
      out[:,4:6] = motion
    else:
      out[:,4:6] = parameters['object_size']

    if not os.path.exists(output_path+'%s/gt/'%(subdrive)):
      os.makedirs(output_path+'%s/gt/'%(subdrive))

    #write MOT formatted gt.txt
    np.savetxt(output_path+'%s/gt/gt.txt'%(subdrive),out,fmt=fmt,delimiter=',')