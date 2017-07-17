#!/usr/bin/env python
"""
This script uses FCN output to infer detection point clusterings
each output file contains the unique detection IDs belonging to a single cluster
"""

import numpy as np
from scipy import misc
from scipy.ndimage.filters import maximum_filter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import os
import argparse
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
parser.add_argument("--fuses",      help="path to fuse files")
parser.add_argument("--images",     help="path to CNN predictions")
parser.add_argument("--output",     help="path to output fuse files")
parser.add_argument("--poses",      help="path to pose CSV files")
parser.add_argument("--config",     help="path to config file")
parser.add_argument("--drives",     help="path to drives list text-file")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

if args.verbosity>=2:
  print(__doc__)

if not os.path.exists(args.output):
  os.makedirs(args.output)

#read preprocessing parameters from the configuration JSON file
with open(args.config) as fparam:
  parameters = json.loads(jsmin(fparam.read()))["gengraph"]

#get names of drives to be processed
drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

for drive in drive_list:
  if args.verbosity>=2:
    print('Working on drive %s'%drive)

  if not os.path.exists(os.path.join(args.output,drive)):
    os.makedirs(os.path.join(args.output,drive))

  #get a list of images for this drive
  image_path = os.path.join(args.images,drive)
  image_list = [i for i in os.listdir(image_path) if i.endswith('png')]

  for image in image_list:
    if os.path.exists(os.path.join(args.output,drive,image)): continue

    if args.verbosity>=2:
      print('Working on image %s'%(image))

    #load the image
    I = misc.imread(os.path.join(args.images,drive,image))/(2.**16-1)

    #read the peak points from the fuse files
    fuse_filename = '%s_%s.png.fuse'%(drive,image.split('_')[0])
    P = np.loadtxt(os.path.join(args.fuses,fuse_filename),delimiter=',')

    #build the affinity matrix
    A = np.tril(squareform(pdist(P[:,3:5])<parameters['distance_threshold']))

    #discard links with gaps along them
    links = np.argwhere(A)
    checkpoints = np.arange(.1,1,.1) #there are 9 checkpoints on each link
    n_checkpoints = len(checkpoints)
    for i,j in links:
      w = np.sum(I[tuple((P[i,3:5]+alpha*(P[j,3:5]-P[i,3:5])).astype(int))] for alpha in checkpoints)
      if w/n_checkpoints<parameters['min_avg_pixel']: A[i,j] = False

    #finding the connected components (trees)
    cc_list = nx.connected_components(nx.from_numpy_matrix(A+A.T)) #list of sets

    os.makedirs(os.path.join(args.output,drive,image[:-4]))
    for group_id,cc in enumerate(cc_list):
      if len(cc)>1:
        out_filename = os.path.join(args.output,drive,image[:-4],'group_%d.fuse'%(group_id))
        np.savetxt(out_filename,[P[i,0] for i in cc],fmt=['%07d'],delimiter=',')
