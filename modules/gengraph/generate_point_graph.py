"""
This script uses FCN output to infer connections between extracted peak points
"""

import numpy as np
from scipy import misc
from scipy.ndimage.filters import maximum_filter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import os, time
import json
from jsmin import jsmin

parser = argparse.ArgumentParser()
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

    if args.verbosity>=2:
      print('Working on image %s'%(image))

    if os.path.exists(os.path.join(args.output,drive,image)):
      continue
    else:
      os.makedirs(os.path.join(args.output,drive,image))

    #load the image
    I = misc.imread(os.path.join(args.images,drive,image))/(2.**16-1)

    peaks = (I==maximum_filter(I,size=10))
    P = np.argwhere(np.logical_and(peaks,I>.4)) #points

    if verbosity>1: print('\tplotting the points')
    ax.plot(P[:,0],P[:,1],'g.')

    if verbosity>1: print('\tbuilding the affinity matrix')
    A = np.tril(squareform(pdist(P)<50))

    if verbosity>1: print('\tdiscarding links with gaps along them')
    links = np.argwhere(A)
    ratios = np.arange(.1,1,.1) #there are 9 checkpoints on each link
    for i,j in links:
      w = np.sum(I[tuple((P[i]+alpha*(P[j]-P[i])).astype(int))] for alpha in ratios) / 9
      if w<.9: A[i,j] = False

    if verbosity>1: print('\tcomputing the minimum spanning forest')
    B = np.tril(nx.to_numpy_matrix(nx.minimum_spanning_tree(nx.from_numpy_matrix(A+A.T))).astype(bool))

    if verbosity>1: print('\tfinding the connected components (trees)')
    cc_list = nx.connected_components(nx.from_numpy_matrix(B+B.T)) #list of sets

    if verbosity>1: print('\tplotting lane boundaries')
    mask = np.zeros_like(B,dtype=bool)
    for cc in cc_list:
      if len(cc)>1:
        mask[...] = False; mask[np.ix_(list(cc),list(cc))] = True
        ax.add_collection(LineCollection(([P[i],P[j]] for i,j in np.argwhere(np.logical_and(B,mask))),color=np.random.rand(3)))

    if verbosity>1: print('\tsaving the figure')
    plt.savefig('out/%s'%(image))

    if verbosity>1: print('\ttook %f seconds for this image'%(time.time()-t0))