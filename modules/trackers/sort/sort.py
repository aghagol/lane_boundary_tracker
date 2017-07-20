#!/usr/bin/env python
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import argparse
import json
from jsmin import jsmin
from filterpy.kalman import KalmanFilter


def d2t_dist(z, x):  # detection to track similarity
    """
    Computes distance between a detection (2x1 numpy array) and a prediction (2x1 numpy array)
    """
    y = (z - x[:2]).squeeze()  # vector from detection to prediction (displacement)
    v = x[2:].squeeze()  # velocity (motion) vector
    d = y - (y.dot(v) / v.dot(v)) * v if abs(v).sum() > 0 else y
    return d.dot(d)


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects
    """
    count = 0

    def __init__(self, initial_state=np.zeros((4, 1)), det_idx=0, det_time=None, motion_model_var=1, observation_var=1):
        """
        Initialises a tracker
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.Q[2:, 2:] *= motion_model_var  # model-induced motion variance - default 0.01
        self.kf.R *= observation_var  # observation variance/uncertainty - default 10
        self.kf.x = initial_state
        self.age_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.confidence = .1
        self.ret = initial_state[:2]
        self.det_idx = det_idx
        self.det_time = det_time
        self.det_x = initial_state[:2]

    def update(self, target_location, det_idx=0, det_time=None):
        """
        Updates the state vector with observations
        """
        self.age_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.confidence = 1.
        self.kf.update(target_location)
        self.ret = self.kf.x[:2]
        self.det_idx = det_idx
        self.det_time = det_time
        self.det_x = target_location

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if (self.age_since_update > 0):
            self.hit_streak = 0
        self.age_since_update += 1
        self.confidence *= .95
        self.ret = self.kf.x[:2]
        self.det_idx = 0
        return self.kf.x


def associate_detections_to_trackers(detections, trackers, d2t_dist_thresh):
    """
    Assigns detections (d x 2 numpy array) to tracked object (t x 2 numpy array)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 2), dtype=int)

    d2t_dist_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            d2t_dist_matrix[d, t] = d2t_dist(det.reshape(2, 1), trk.reshape(4, 1))

    matched_detections, matched_tracks = [i.flatten() for i in linear_sum_assignment(d2t_dist_matrix)]

    unmatched_detections = np.delete(np.arange(len(detections)), matched_detections).tolist()
    unmatched_trackers = np.delete(np.arange(len(trackers)), matched_tracks).tolist()

    # filter out matched with high d2t_dist
    matches = []
    for d, t in zip(matched_detections, matched_tracks):
        if (d2t_dist_matrix[d, t] > d2t_dist_thresh[t]):
            unmatched_detections.append(d)
            unmatched_trackers.append(t)
        else:
            matches.append([d, t])
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self,
                 max_age_since_update=1,
                 max_mov_since_update=1,
                 min_hits=3,
                 d2t_dist_threshold_tight=10,
                 d2t_dist_threshold_loose=10,
                 motion_model_variance=1,
                 observation_variance=1,
                 motion_init_pose=False,
                 motion_init_sync=False,
                 ):
        """
        Sets key parameters for SORT
        """
        self.trackers = []
        self.frame_count = 0
        self.time_gap = 0
        self.time_now = 0
        self.max_age_since_update = max_age_since_update
        self.max_mov_since_update = max_mov_since_update
        self.min_hits = min_hits
        self.d2t_dist_threshold_tight = d2t_dist_threshold_tight
        self.d2t_dist_threshold_loose = d2t_dist_threshold_loose
        self.motion_model_variance = motion_model_variance
        self.observation_variance = observation_variance
        self.motion_init_pose = motion_init_pose
        self.motion_init_sync = motion_init_sync

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,u,v,idx,time],[x,y,u,v,idx,time],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # compute time step from the last update until now
        self.time_gap = dets[0, 5] - self.time_now if self.frame_count > 1 else 0

        # update the current time
        self.time_now = dets[0, 5]

        # add a tiny amount (eps=0.1 microseconds) to time_gap to prevent numerical instability
        self.time_gap += 1e-7

        # get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))  # columns: x,y,velocity_x,velocity_y
        to_del = []

        for t, trk in enumerate(trks):
            self.trackers[t].kf.F = np.array(
                [[1, 0, self.time_gap, 0], [0, 1, 0, self.time_gap], [0, 0, 1, 0], [0, 0, 0, 1]])
            # self.trackers[t].kf.Q *= self.time_gap #more uncertainty for large time steps
            target_state = self.trackers[t].predict()
            trk[:] = target_state.squeeze()
            if (np.any(np.isnan(target_state))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in to_del[::-1]:
            self.trackers.pop(t)

        # associate detections with trackers
        # EDIT ------------ tighten d2t_dist_theshold after the first hit (mohammad)
        d2t_dist_thresh = []
        for t, trk in enumerate(self.trackers):
            if trk.hits > 0:
                d2t_dist_thresh.append((self.d2t_dist_threshold_tight) ** 2)
            else:
                d2t_dist_thresh.append((self.d2t_dist_threshold_loose) ** 2)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets[:, :2], trks, d2t_dist_thresh)
        # -------------------------------------------------------

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :2].reshape(2, 1), det_idx=dets[d, 4], det_time=self.time_now)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:

            # generate the initial state for new trackers
            if self.motion_init_pose:
                # use pose-based estimated motion
                initial_state = dets[i, :4].reshape(4, 1)
            elif self.motion_init_sync:
                # compute average motion of matched tracks
                tracker_states = [np.array(trk.kf.x) for trk in self.trackers if trk.hits > 0]
                if len(tracker_states) > 0:
                    avg_velocity = np.array(tracker_states).mean(axis=0)[2:].reshape(2, 1)
                else:
                    avg_velocity = np.zeros((2, 1))
                initial_state = np.concatenate((dets[i, :2].reshape(2, 1), avg_velocity))
            else:
                initial_state = np.concatenate((dets[i, :2].reshape(2, 1), np.zeros((2, 1))))

            trk = KalmanBoxTracker(
                initial_state=initial_state,
                det_idx=dets[i, 4],
                det_time=self.time_now,
                motion_model_var=self.motion_model_variance,
                observation_var=self.observation_variance)
            self.trackers.append(trk)

        # output tracking results
        ret = []
        for trk in self.trackers:
            d = trk.ret.squeeze()  # customized return value
            if trk.det_idx:  # only output if matched with an observation (trk.det_idx will be zero otherwise)
                ret_item = []
                ret_item.append(trk.id + 1)  # target id (+1 as MOT benchmark requires positive)
                ret_item.append(d[0])  # x location
                ret_item.append(d[1])  # y location
                ret_item.append(trk.det_idx)  # unique detection id
                ret_item.append(trk.confidence)  # tracking confidence
                ret.append(np.array(ret_item).reshape((1, -1)))

        # remove dead tracklets
        for t in range(len(self.trackers) - 1, -1, -1):
            dead_flag_age = (self.trackers[t].age_since_update >= self.max_age_since_update)
            dead_flag_mov = (
            ((self.trackers[t].det_x - self.trackers[t].kf.x[:2]) ** 2).sum() > self.max_mov_since_update ** 2)
            if dead_flag_mov or dead_flag_age:
                self.trackers.pop(t)

        if (len(ret) > 0):
            return np.concatenate(ret)
        return ret


def run(input, output, config, verbosity):
    total_time = 0.0
    total_frames = 0

    with open(config) as fparam:
        param = json.loads(jsmin(fparam.read()))["sort"]

    if not os.path.exists(output):
        os.makedirs(output)

    sequences = os.listdir(input)
    for seq in sequences:
        if os.path.exists('%s/%s.txt' % (output, seq)): continue

        mot_tracker = Sort(
            max_age_since_update=param['max_age_after_last_update'],
            max_mov_since_update=param['max_mov_after_last_update'],
            d2t_dist_threshold_tight=param['d2t_dist_threshold_tight'],
            d2t_dist_threshold_loose=param['d2t_dist_threshold_loose'],
            motion_model_variance=param['motion_model_variance'],
            observation_variance=param['observation_variance'],
            motion_init_pose=param['motion_init_pose'],
            motion_init_sync=param['motion_init_sync'],
        )  # create instance of the SORT tracker

        if param['bypass_tracking_use_gt']:
            seq_points = np.loadtxt('%s/%s/gt/gt.txt' % (input, seq), delimiter=',')  # load detections
            if verbosity >= 2:
                print('\tWARNING: using ground-truth data (no tracking)')
        else:
            seq_points = np.loadtxt('%s/%s/det/det.txt' % (input, seq), delimiter=',')  # load detections

        seq_points[:, 7] *= 1e-6  # convert micro-seconds to seconds

        if verbosity >= 2:
            print("\nProcessing %s" % (seq))

        start_time = time.time()
        with open('%s/%s.txt' % (output, seq), 'w') as out_file:
            for frame_idx, frame in enumerate(sorted(set(seq_points[:, 0]))):
                points = seq_points[seq_points[:, 0] == frame, :]
                if param['bypass_tracking_use_gt']:
                    for d in points:
                        print('%05d,%05d,%011.5f,%011.5f,%05d,1.00' % (frame, d[1], d[2], d[3], d[6]), file=out_file)
                else:
                    # input points format: x, y, x_vel, y_vel, det_idx, timestamp
                    tracks = mot_tracker.update(points[:, 2:8])
                    for d in tracks:
                        # output tracking format: frame number, target id, x, y, detection id, condifdence
                        print('%05d,%05d,%011.5f,%011.5f,%07d,%.2f' % (frame, d[0], d[1], d[2], d[3], d[4]),
                              file=out_file)

        total_time += time.time() - start_time
        total_frames += seq_points.shape[0]
    if verbosity >= 2:
        print("Total Tracking took: %.3f for %d frames" % (total_time, total_frames))


def main(argv):
    parser = argparse.ArgumentParser(description='SORT Tracker')
    parser.add_argument("--input", help="path to MOT dataset")
    parser.add_argument("--output", help="output path to save tracking results")
    parser.add_argument("--config", help="configuration JSON file")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    run(args.input, args.output, args.config, args.verbosity)

if __name__ == '__main__':
    main(sys.argv[1:])