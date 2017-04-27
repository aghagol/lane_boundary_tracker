"""
This is a 2-pass algorithm for fusing/stitching tracklets 
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import networkx as nx

def fuse(tracks,param):
	#Each track is associated with a lane boundary (second column in tracks array)
	#Assign each lane boundary (lb) to a node in the track affinity graph
	lb2node = {lb:node for node,lb in enumerate(sorted(set(tracks[:,1])))}
	node2lb = {node:lb for lb,node in lb2node.iteritems()}

	#Compute track-to-track affinities
	A = np.zeros((len(lb2node),len(lb2node)),dtype=int)
	frames = np.unique(tracks[:,0])
	for frame_idx,frame in enumerate(frames[:-1]):
		active_tracks = tracks[np.logical_and(tracks[:,0]>=frame,tracks[:,0]<=frames[frame_idx+1]),:]
		active_nodes_pdist = squareform(pdist(active_tracks[:,2:4],metric='euclidean'))
		active_nodes = [lb2node[lb] for lb in active_tracks[:,1]] #I admit there will be duplicates in active_nodes
		A[np.ix_(active_nodes,active_nodes)] += (active_nodes_pdist<param['gating_thresh'])
	np.fill_diagonal(A,0) #no self-loops allowed

	#Mark tracks for fusion by finding connected components of the tracklets graph:
	G = nx.from_numpy_matrix(np.logical_and(A>=param['affinity_thresh_min'],A<=param['affinity_thresh_max']))
	node_label = {node:k for k,comp_set in enumerate(nx.connected_components(G)) for node in comp_set}
	tracks[:,1] = map(lambda lb: node_label[lb2node[lb]]+1,tracks[:,1]) #WARNING: over-writing lb values

	#Use tracking confidence scores to remove overlaps (keep points with highest confidences)
	out = []
	for frame in range(1,int(tracks[:,0].max())+1):
		active_tracks = tracks[tracks[:,0]==frame,:]
		for label in set(active_tracks[:,1]):
			current_track = active_tracks[active_tracks[:,1]==label,:]
			if current_track.shape[0]>1: #checking for overlaps
				out.append(current_track[np.argmax(current_track[:,-1]),:])
			else:
				out.append(current_track)
	out = np.vstack(out)
	return out