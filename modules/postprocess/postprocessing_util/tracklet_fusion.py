"""
This is a 2-pass algorithm for fusing/stitching tracklets 
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def fuse(tracks,param):
	#Each track is associated with a lane boundary (second column in tracks array)
	#Assign each lane boundary (lb) to a node in the track affinity graph
	lb2node = {lb:node for node,lb in enumerate(sorted(set(tracks[:,1])))}
	node2lb = {node:lb for lb,node in lb2node.iteritems()}

	#Compute track-to-track affinities
	A = np.zeros((len(lb2node),len(lb2node)),dtype=int)
	for frame in range(1,int(tracks[:,0].max())+1):
		active_tracks = tracks[tracks[:,0]==frame,:]
		active_nodes = [lb2node[lb] for lb in active_tracks[:,1]]
		active_nodes_pdist = squareform(pdist(active_tracks[:,2:4],metric='euclidean'))
		A[np.ix_(active_nodes,active_nodes)] += (active_nodes_pdist<param['gating_thresh'])
	np.fill_diagonal(A,0) #no self-loops allowed

	#Mark tracks for fusion (assign a common label to multiple tracks)
	#I use a region growing approach seeding from node 1
	#=find connected components of graph (for now)
	node_label = {node:node for node in node2lb}
	for node_1 in node2lb:
		for node_2 in node2lb:
			if A[node_1,node_2]>=param['affinity_thresh']:
				new_label = min(node_label[node_1],node_label[node_2])
				node_label[node_1] = new_label
				node_label[node_2] = new_label
	tracks[:,1] = [node_label[lb2node[lb]] for lb in tracks[:,1]] #WARNING: over-riding lb names

	#Use tracking confidence scores to remove the overlap
	out = []
	for frame in range(1,int(tracks[:,0].max())+1):
		active_tracks = tracks[tracks[:,0]==frame,:]
		for label in set(active_tracks[:,1]):
			current_track = active_tracks[active_tracks[:,1]==label,:]
			if current_track.shape[0]>1: #check for overlaps
				#keep the tracked point with highest confidence
				out.append(current_track[np.argmax(current_track[:,6]),:])
			else:
				out.append(current_track)
	out = np.vstack(out)
	return out