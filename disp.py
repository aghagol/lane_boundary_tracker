from __future__ import print_function
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
from sort import Sort

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
F = 100 # last frame to plot

plt.ion()
# plt.ioff()
for seq in sequences:
  fig, ax = plt.subplots(1,1)
  dets_all = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
  dets = dets_all[dets_all[:,6]>.5,:] # filter based on detection confidence
  tracks = np.loadtxt('output/%s.txt'%(seq),delimiter=',')
  ax.set_xlim([dets[:,2].min(),dets[:,2].max()])
  ax.set_ylim([dets[:,3].min(),dets[:,3].max()])
  for frame in range(F):
    ax.plot(dets[dets[:,0]==frame,2],dets[dets[:,0]==frame,3],'k.',markersize=1)
    ax.set_title('frame %d'%frame)
    plt.pause(.1)
    # fig.canvas.flush_events()
    # plt.draw()
    # ax.cla()
    # ax.cla()
  # plt.savefig('out.png',ppi=300)






















