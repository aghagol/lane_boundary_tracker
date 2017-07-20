"""
This is an algorithm for reducing the number of points in each track
Input is a numpy matrix with each row's format as: frame_id, target_id, x, y, detection_id, confidence
Output is a numpy matrix with the same format as input
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from rdp import rdp


def reducer(tracks, param):
    rdp_mask = np.zeros(tracks.shape[0], dtype=bool)
    for target_id in set(tracks[:, 1]):
        label_mask = (tracks[:, 1] == target_id)  # put points with same label together
        rdp_mask[label_mask] = rdp(tracks[label_mask, 2:4], epsilon=param['rdp_epsilon'], return_mask=True)
    return tracks[rdp_mask, :]
