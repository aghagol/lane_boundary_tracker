import numpy as np
from scipy import misc
from scipy.ndimage.filters import maximum_filter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import os, time

verbosity = 1
drive_id = 'HT073_1479214910'

image_path = os.path.join('.','images',drive_id)
image_list = [i for i in os.listdir(image_path) if i.endswith('png')]

output_path = 'out'
if not os.path.exists(output_path):
  os.makedirs(output_path)

fig,ax = plt.subplots(1,1,figsize=(50,50))
for image in image_list:

  print('Working on image %s'%(image))
  if os.path.exists('out/%s'%(image)):
    print('\toutput exists... skipping')
    continue

  ax.cla()
  t0 = time.time()

  if verbosity>1: print('\tloading the image')
  I = misc.imread(os.path.join(image_path,image))/(2.**16-1)
  ax.imshow(I.T,cmap='gray',vmin=0,vmax=2)

  if verbosity>1: print('\textracting the points')
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