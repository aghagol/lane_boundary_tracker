#!/usr/bin/env python
"""
This script creates an MOT-like dataset from timestamp tagged points
"""

import os
from PIL import Image
import numpy as np
import argparse
import json
from jsmin import jsmin
import networkx as nx

import motutil

parser = argparse.ArgumentParser()
parser.add_argument("--tagged",     help="path to tagged annotations")
parser.add_argument("--output",     help="output path to MOT dataset")
parser.add_argument("--config",     help="path to config file")
parser.add_argument("--drives",     help="path to drives list file")
parser.add_argument("--poses",      help="path to drive pose CSV files")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

if args.verbosity>=2:
  print(__doc__)

with open(args.config) as fparam:
  param = json.loads(jsmin(fparam.read()))["preprocess"]

drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

#for each drive generate a sequence (or a series of sequence in the presence of long time gaps)
for drive in drive_list:
  if args.verbosity>=2:
    print('Working on drive %s'%drive)

  #compute a list of fuse files (one for each image) that belong to the current drive
  filelist = sorted([ i for i in os.listdir(args.tagged)
    if ('_'.join(i.split('_')[:2])==drive and i.endswith('.fuse') and os.stat(args.tagged+'/'+i).st_size) ])

  #if split_on_temporal_gaps option is enabled, break the drive into smaller parts in the presence of long time gaps
  if param['split_on_temporal_gaps']:
    if args.verbosity>=2:
      print('\tClustering images based on time overlap...')

    #read image stats
    image_stat = {}
    for filename in filelist:
      #points format: id, latitude, longitude, altitude, timestamp
      points = np.loadtxt(args.tagged+'/'+filename,delimiter=',').reshape(-1,5)
      image_stat[filename] = (points[:,4].min()*1e-6,points[:,4].max()*1e-6,points.shape[0])

    #build an "interval graph" based on image stats
    A = np.zeros((len(filelist),len(filelist)),dtype=bool)
    for file_idx1,filename1 in enumerate(filelist):
      for file_idx2,filename2 in enumerate(filelist):
        flag1 = image_stat[filename2][0]-image_stat[filename1][0]<param['gap_min']
        flag2 = image_stat[filename1][0]-image_stat[filename2][1]<param['gap_min']
        flag3 = image_stat[filename2][0]-image_stat[filename1][1]<param['gap_min']
        flag4 = image_stat[filename1][1]-image_stat[filename2][1]<param['gap_min']
        A[file_idx1,file_idx2] = (flag1 and flag2) or (flag3 and flag4)

    #find connected components of the interval graph
    cc_list = nx.connected_components(nx.from_numpy_matrix(A)) #returns list of sets

    #map the node indices in connected components to fuse filenames
    cc_list = [[filelist[file_idx] for file_idx in sorted(cc)] for cc in cc_list]

    #compute number of points accumulated in each connected component
    cc_n = [np.sum([image_stat[filename][2] for filename in cc]) for cc in cc_list]
    
    #build a dictionary of drive parts
    clusters = {'%s_part_%05d'%(drive,k):cc for k,cc in enumerate(cc_list) if cc_n[k]>=param['min_seq_size']}

    #print useful info
    for i,cc in enumerate(cc_list):
      if cc_n[i]<param['min_seq_size']:
        if args.verbosity>=2:
          print('\t...discarding %s_part_%05d due to small size (%d<%d)'%(drive,i,cc_n[i],param['min_seq_size']))
      else:
        if args.verbosity>=2:
          print('\t...subdrive %s_part_%05d has the following members:'%(drive,i))
        for filename in cc:
          if args.verbosity>=2:
            print('\t\t%s'%(filename))
  else:
    #process all images as part of a single sequence (for the entire drive)
    clusters = {drive:filelist}

  #keep track of subdrives with 0 or 1 detection points that will be marked for deletion
  #these subdrives go unseen until now because points can be removed in later stages
  tiny_subdrives = set()

  #create initial sequences in ITLLAL format (detection index,timestamp,latitude,longitude,altitude,GT label)
  if args.verbosity>=2:
    print('\tCreating ITLLAL.txt')
  motutil.generate_ITLLAL_and_tmap(args.tagged,args.output,clusters,tiny_subdrives,param)

  for subdrive in tiny_subdrives:
    if args.verbosity>=2:
      print('\tMarking %s for deletion due to insufficient points!'%(subdrive))

  #create MOT formated det.txt files
  if args.verbosity>=2:
    print('\tCreating MOT det.txt')
  motutil.generate_MOT_det(args.output,clusters,tiny_subdrives,args.poses+'/'+drive+'-pose.csv',param)

  #save (fake) images for MOT compatibility
  if param['generate_fake_images']:
    if args.verbosity>=2:
      print('\tCreating MOT (fake) images')
    for subdrive in clusters:
      img_out = args.output+'/%s/img1/'%(subdrive)
      if not os.path.exists(img_out):
        os.makedirs(img_out)
      Image.fromarray(np.zeros((1,1))).convert('RGB').save(img_out+'000001.jpg')
