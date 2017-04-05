#!/usr/bin/env python

import os, re
import numpy as np

stat_path = 'stat_out'

#populate stats if not already there
if not os.path.exists(stat_path):
  os.mkdir(stat_path)
  for name in os.listdir('data'):
#    print('Working on %s'%(name))
    os.system("python pymot/pymot.py -a out/%s/groundtruth.json -b out/%s/hypotheses.json > %s/%s.stat"%(name,name,stat_path,name))

#parse stat files
stat_list = [i for i in os.listdir(stat_path) if i.endswith('stat')]

stat = {}
for stat_file in stat_list:
  with open(stat_path+'/'+stat_file) as f:
    stat_str = f.read()
  stat[stat_file] = {k.strip():v for k,v in re.findall('(.+[^\d+\.{0,1}\d*])(\d+\.{0,1}\d*)',stat_str)}

print('Total Stats:')
print('...Number of surfaces : %d'%(len(stat)))

n_gt = np.sum([int(i['Ground truths']) for i in stat.values()])
print('...Number of GT points : %d'%(n_gt))

n_gt_tracks = np.sum([int(i['Total ground truth tracks']) for i in stat.values()])
print('...Number of GT tracks : %d'%(n_gt_tracks))

n_hyp_tracks = np.sum([int(i['Total hypothesis tracks']) for i in stat.values()])
print('...Number of predicted tracks : %d'%(n_hyp_tracks))

n_fn = np.sum([int(i['Misses']) for i in stat.values()])
print('...FN : %d'%(n_fn))

n_fp = np.sum([int(i['False positives']) for i in stat.values()])
print('...FP : %d'%(n_fp))

mota = np.mean([float(i['MOTA']) for i in stat.values()])
print('...MOTA (avg) : %f'%(mota))

motp = np.mean([float(i['MOTP']) for i in stat.values()])
print('...MOTP (avg): %f'%(motp))
