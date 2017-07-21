"""
This is an algorithm for fusing tracking results with image-based clustering results
Input 1 is a numpy matrix with each row's format as: frame_id, target_id, x, y, detection_id, confidence
Input 2 is a numpy matrix with each row's format as: group_id, detection_id
Output is a numpy matrix with the same format as input
"""
import numpy as np

def fusion(tracks,groups,param):
  det_group = dict(groups)
  for target_id in set(tracks[:,1]):
    target_points = tracks[tracks[:,1]==target_id,:]
    target_groups = [det_group[det] for det in target_points[:,4] if det in det_group]
  return tracks