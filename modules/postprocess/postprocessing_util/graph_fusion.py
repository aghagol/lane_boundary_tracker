"""
This is an algorithm for fusing tracking results with image-based clustering results
Input 1 is a numpy matrix with each row's format as: frame_id, target_id, x, y, detection_id, confidence
Input 2 is a numpy matrix with each row's format as: group_id, detection_id
Output is a numpy matrix with the same format as input
"""
import numpy as np
import networkx as nx
import ipdb

class track:

  def __init__(self, track_id, head_idx, tail_idx, head_group, tail_group):

    self.id = track_id
    self.head_idx = head_idx
    self.tail_idx = tail_idx
    self.head_group = head_group
    self.tail_group = tail_group

def fusion(tracks,groups,param):

  #a dictionary of {dettection_id: group_label}
  det_group = dict(groups)

  #list of track objects
  track_list = []
  track_id_list = sorted(set(tracks[:,1]))

  head_info = {}
  tail_info = {}

  for track_id in track_id_list:

    #list of indices (sorted temporally) associated with this track
    idx_list = np.argwhere(tracks[:,1]==track_id).astype(int)

    #compute head and tail indices and their corresponding group labels
    head_idx = idx_list.min()
    head_group = det_group[tracks[head_idx, 4]]
    tail_idx = idx_list.max()
    tail_group = det_group[tracks[tail_idx, 4]]

    head_info[track_id] = head_idx
    tail_info[track_id] = tail_idx

    #append this track object to track list
    track_list.append(track(track_id, head_idx, tail_idx, head_group, tail_group))

  #find pairs of tracks to be merged
  A = np.zeros((len(track_list), len(track_list)), dtype=bool)
  for i1,trk1 in enumerate(track_list):
    for i2,trk2 in enumerate(track_list):
      if (trk1.tail_idx < trk2.head_idx) and (trk1.tail_group > 0) and (trk1.tail_group == trk2.head_group):
        A[i1, i2] = True
        A[i2, i1] = True

  #no junctions, loops, ... allowed vvv
  A = np.array(nx.to_numpy_matrix(nx.minimum_spanning_tree(nx.from_numpy_matrix(A))).astype(bool))
  hubs = A.sum(axis=1) > 2
  A[hubs, :] = False
  A[:, hubs] = False

  cc_list = nx.connected_components(nx.from_numpy_matrix(A))  #list of sets
  for cc in cc_list:
    if len(cc)>2:
      for i in cc:
        neighbors = np.argwhere(A[i]).flatten().tolist()
        if len(neighbors)>1:
          if len(neighbors)>2:
            exit('something is off!')
          head_0 = head_info[track_id_list[neighbors[0]]]
          head_1 = head_info[track_id_list[neighbors[1]]]
          head_i = head_info[track_id_list[i]]
          if (head_0 > head_i and head_1 > head_i) or (head_0 < head_i and head_1 < head_i):
            A[i, neighbors[0]] = False
            A[neighbors[0], i] = False
            A[i, neighbors[1]] = False
            A[neighbors[1], i] = False
  cc_list = nx.connected_components(nx.from_numpy_matrix(A))  #list of sets

  track_label_map = {track_id_list[i]:(label+1) for label, cc in enumerate(cc_list) for i in cc}

  #assign same (new) track IDs to tracks that must be merged
  tracks[:,1] = map(lambda x: track_label_map[x], tracks[:,1])

  return tracks