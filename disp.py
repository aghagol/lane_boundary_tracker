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
  # 'TUD-Campus',
  # 'TUD-Stadtmitte',
  # 'ETH-Bahnhof',
  # 'ETH-Sunnyday',
  # 'ETH-Pedcross2',
  # 'KITTI-13',
  # 'KITTI-17',
  # 'ADL-Rundle-6',
  # 'ADL-Rundle-8',
  # 'Venice-2',
]

colours = np.random.rand(32,3) #used only for display
F = range(700) # list of frames to plot
margin = 50 # add pixels to plot margins

try:
  fig, ax = plt.subplots(1,1)
  for seq in sequences:
    print('Working on sequence %s'%seq)
    # dets_all = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
    dets_all = np.loadtxt('mot_benchmark/%s/%s/det/det.txt'%(phase,seq),delimiter=',')
    dets = dets_all[dets_all[:,6]>.5,:6] # filter based on detection confidence
    dets[:,4:6] += dets[:,2:4] # convert from [x1,y1,w,h] to [x1,y1,x2,y2]
    for i in range(dets.shape[0]):
      dets[i,2:6] = np.squeeze(convert_bbox_to_z(dets[i,2:6]))
    tracks = np.loadtxt('output/%s.txt'%(seq),delimiter=',')
    for frame in F:
      ax.cla()
      ax.set_xlim([dets[:,2].min()-margin,dets[:,2].max()+margin])
      ax.set_ylim([dets[:,3].min()-margin,dets[:,3].max()+margin])
      ax.plot(dets[dets[:,0]==frame,2],dets[dets[:,0]==frame,3],'ko')
      ax.set_title('frame %d'%frame)
      plt.pause(.01)
except KeyboardInterrupt:
  exit('')





















