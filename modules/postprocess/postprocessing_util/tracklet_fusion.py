"""
This is a 2-pass algorithm for fusing/stitching tracklets 
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def fuse(trks,param):
	trk_id_id = {trk_id:id_id for id_id,trk_id in enumerate(sorted(set(trks[:,1])))}

	#compute track-to-track affinities
	affinity = np.zeros((len(trk_id_id),len(trk_id_id)),dtype=int)
	for frame in range(1,int(trks[:,0].max())+1):
		trks_active = trks[trks[:,0]==frame,:]
		trks_pdist = squareform(pdist(trks_active[:,2:4],metric='euclidean'))
		ids = [trk_id_id[trk_id] for trk_id in trks_active[:,1]]
		affinity[np.ix_(ids,ids)] += (trks_pdist<param['gating_thresh'])
	np.fill_diagonal(affinity,0)

	#assign a common track id to tracks that should get fused
	#here I use a recursive algorithm to assign the smallest id to matched tracks
	id_map = {trk_id:trk_id for trk_id in trk_id_id}
	for trk_id_1,id_id_1 in trk_id_id.iteritems():
		for trk_id_2,id_id_2 in trk_id_id.iteritems():
			if affinity[id_id_1,id_id_2]>param['affinity_thresh']:
				id_map[trk_id_1] = min(id_map[trk_id_1],id_map[trk_id_2])
				id_map[trk_id_2] = min(id_map[trk_id_1],id_map[trk_id_2])

	#use tracking confidence scores to remove weaker tracks (duplicates)
	out = []
	for frame in range(1,int(trks[:,0].max())+1):
		trks_active = trks[trks[:,0]==frame,:]
		for trk_id in set(trks_active[:,1]):
			tmp = trks_active[trks_active[:,1]==trk_id,:]
			if tmp.shape[0]>1: #check for duplicates
				out.append(tmp[np.argmax(tmp[:,6]),:])
			else:
				out.append(tmp)
	out = np.vstack(out)
	return out