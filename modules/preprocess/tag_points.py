#!/usr/bin/env python
"""
This script assigns timestamps to detections
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
from jsmin import jsmin

import motutil


def run(fuses, tagged, images, config, drives, poses, verbosity):
    if not os.path.exists(tagged):
        os.makedirs(tagged)

    with open(config) as fparam:
        parameters = json.loads(jsmin(fparam.read()))["preprocess"]

    drive_list = []
    with open(drives) as fdrivelist:
        for line in fdrivelist:
            drive_list.append(line.strip())

    # final format for .fuse files with timestmaps
    # format: id, latitude, longitude, altitude, timestamp
    tag_fmt = ['%07d', '%.10f', '%.10f', '%.10f', '%016d']

    for drive in drive_list:
        if verbosity >= 2:
            print('Working on drive %s' % drive)

        # read and process drive pose
        if os.path.isdir(poses):
            pose_path = poses + '/' + drive + '-pose.csv'
        else:
            pose_path = poses
        pose = np.loadtxt(pose_path)  # pose points format: latitude, longitude, altitude, timestamp

        # re-arrange pose points (rows) based on their timestamps
        pose = pose[pose[:, 3].argsort(), :]

        # convert pose point coordinates from lat-lon to meters of distance from the origin (bottom left corner)
        # also, store the scaling parameters for meterization of detection points
        pose_meterized, scale_meta = motutil.meterize(pose)

        # generate a mapping from the original timestmaps to "fake" timestamps (in microseconds)
        # the interval between each 2 consecutive pose points will be divided into 1e6 microseconds
        pose_tmap = {original_timestamp: counter * 1e6 for counter, original_timestamp in enumerate(pose[:, 3])}

        # get meta information (in CSV format) about topdown images
        # meta column labels: name, time_start, time_end, min_lat, min_lon, max_lat, max_lon
        meta = pd.read_csv(os.path.join(images, drive, 'meta.csv'), skipinitialspace=True)

        # get the list of all fuse files (containing detection points) for this drive
        filelist = sorted([i for i in os.listdir(fuses) if '_'.join(i.split('_')[:2]) == drive])

        for filename in filelist:

            # skip timestamp generation if the current image has been processed before
            if os.path.exists(os.path.join(tagged, filename)) and os.path.exists(
                            os.path.join(tagged, filename) + '.tmap') >= parameters['fake_timestamp']:
                if verbosity >= 2:
                    print('\t%s exists! skipping' % (tagged + '/' + filename))
                continue

            if verbosity >= 2:
                print('\tworking on %s' % (os.path.join(fuses, filename)))

            # read the detection points for the current image
            points = np.loadtxt(os.path.join(fuses, filename), delimiter=',').reshape(-1, 5)

            # clip the pose path according to the current image lat-lon bounds
            # this is how it works:
            # 1 - start from a "seed" point in the pose seq around the center of the image
            # 2 - scan the pose seq by going back in time until an image edge is reached
            # 3 - scan the pose seq by going forward in time until an image edge is reached
            # 4 - clip parts of the pose seq that lie outside of the image
            if parameters['pose_filter_bbox']:
                image_tag = int(filename.split('_')[-1].split('.')[0])
                image_idx = list(meta['time_start']).index(image_tag)
                min_lat = meta['min_lat'][image_idx]
                min_lon = meta['min_lon'][image_idx]
                max_lat = meta['max_lat'][image_idx]
                max_lon = meta['max_lon'][image_idx]
                seed_idx = np.argmax(pose[:, 3] > (meta['time_start'][image_idx] + meta['time_end'][image_idx]) / 2)
                edge_idx = []
                for step in [-1, 1]:
                    i = seed_idx
                    while pose[i, 0] > min_lat and pose[i, 0] < max_lat and pose[i, 1] > min_lon \
                            and pose[i, 1] < max_lon:
                        i += step
                        if i < 0 or i >= len(pose): break
                    edge_idx.append(i - step)
                pose_filtered = pose_meterized[edge_idx[0]:edge_idx[1], :]
            else:
                pose_filtered = pose_meterized

            # WARNING: points array is modified in place (during conversion from lat-lon to meters)
            tagged_res, tagged_tmap = motutil.get_tagged(points, pose_filtered, pose_tmap, scale_meta, parameters)

            # save extended fuse files (augmented with timestamps)
            np.savetxt(os.path.join(tagged, filename), tagged_res, fmt=tag_fmt, delimiter=',')

            # save the mapping between original timestamps and the fake timestamps
            if parameters['fake_timestamp']:
                np.savetxt(os.path.join(tagged, filename) + '.tmap', tagged_tmap, fmt=['%016d', '%016d'],
                           delimiter=',')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuses", help="path to fuse files")
    parser.add_argument("--tagged", help="path to tagged annotations")
    parser.add_argument("--images", help="path to image metadata")
    parser.add_argument("--config", help="path to config file")
    parser.add_argument("--drives", help="path to drives list file")
    parser.add_argument("--poses", help="path to drive pose CSV files")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    if args.verbosity >= 2:
        print(__doc__)

    run(args.fuses, args.tagged, args.images, args.config, args.drives, args.poses,
        args.verbosity)


if __name__ == '__main__':
    main(sys.argv[1:])
