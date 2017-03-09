from __future__ import print_function
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
from sort import Sort
from sort import convert_bbox_to_z
import warnings
warnings.filterwarnings("ignore")

phase = 'train'
sequences = [
  'PETS09-S2L1',
  'TUD-Campus',
  'TUD-Stadtmitte',
  'ETH-Bahnhof',
  'ETH-Sunnyday',
  'ETH-Pedcross2',
  'KITTI-13',
  'KITTI-17',
  'ADL-Rundle-6',
  'ADL-Rundle-8',
  'Venice-2',
]

colours = np.random.rand(711,3) #used only for display
margin = 50 # add pixels to plot margins

fig, ax = plt.subplots(1,1)
for seq in sequences:
  print('Working on sequence %s'%seq)
  try:
    # dets_all = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
    dets_all = np.loadtxt('mot_benchmark/%s/%s/det/det.txt'%(phase,seq),delimiter=',')
    dets = dets_all[dets_all[:,6]>.5,:6] # filter based on detection confidence
    dets[:,4:6] += dets[:,2:4] # convert from [x1,y1,w,h] to [x1,y1,x2,y2]
    for i in range(dets.shape[0]): # convert to [x_center,y_center,scale,ratio]
      dets[i,2:6] = np.squeeze(convert_bbox_to_z(dets[i,2:6]))
    trks = np.loadtxt('output/%s.txt'%(seq),delimiter=',')[:,:6]
    trks[:,4:6] += trks[:,2:4]
    for i in range(trks.shape[0]): # convert to [x_center,y_center,scale,ratio]
      trks[i,2:6] = np.squeeze(convert_bbox_to_z(trks[i,2:6]))
    for frame in range(int(trks[:,0].max())-1):
      ax.cla()
      ax.set_xlim([dets[:,2].min()-margin,dets[:,2].max()+margin])
      ax.set_ylim([dets[:,3].min()-margin,dets[:,3].max()+margin])
      ax.plot(dets[dets[:,0]==frame,2],dets[dets[:,0]==frame,3],'ko')
      trks_active = trks[trks[:,0]==frame,:]
      for trk_alive in trks_active:
        trk_idid = int(trk_alive[1])
        trk_tail = trks[np.logical_and(trks[:,1]==trk_idid,trks[:,0]<=frame),:]
        ax.plot(trk_tail[:,2],trk_tail[:,3],'r',color=colours[trk_idid%32,:])
      ax.set_title('frame %d'%frame)
      plt.pause(.1)
  except KeyboardInterrupt:
    if raw_input(" Press Enter to continue...")=='q': exit('')
    continue





















