#!/usr/bin/env python
"""
This script uses FCN output to infer detection point clusterings
each output file contains the unique detection IDs belonging to a single cluster
"""

import numpy as np
from scipy import misc
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import sys
import os
import argparse
import json
from jsmin import jsmin


def run(fuses, images, output, config, drives, verbosity):
    if not os.path.exists(output):
        os.makedirs(output)

    # read preprocessing parameters from the configuration JSON file
    with open(config) as fparam:
        parameters = json.loads(jsmin(fparam.read()))["gengraph"]

    if not parameters['enable']:
        return

    # get names of drives to be processed
    drive_list = []
    with open(drives) as fdrivelist:
        for line in fdrivelist:
            drive_list.append(line.strip())

    for drive in drive_list:
        if os.path.exists(os.path.join(output, drive) + '.txt'): continue

        if verbosity >= 2:
            print('Working on drive %s' % drive)

        # global group_id counter per drive
        group_id = 1

        # get a list of images for this drive
        image_path = os.path.join(images, drive)
        image_list = [i for i in os.listdir(image_path) if i.endswith('png')]

        for image in image_list:
            #clusters of peak points per image
            clusters = []

            if verbosity >= 2:
                print('Working on image %s' % (image))

            # load the image
            I = misc.imread(os.path.join(images, drive, image), mode='F') / (2 ** 16 - 1)
            I = np.fliplr(I.T)

            # read the peak points from the fuse files
            fuse_filename = '%s_%s.png.fuse' % (drive, image.split('_')[0])
            P = np.loadtxt(os.path.join(fuses, fuse_filename), delimiter=',').astype(int)

            # build the affinity matrix
            A = np.tril(squareform(pdist(P[:, 3:5]) < parameters['distance_threshold']))

            # discard links with gaps along them
            links = np.argwhere(A)
            checkpoints = np.arange(.1, 1, .1)  # there are 9 checkpoints on each link
            n_checkpoints = len(checkpoints)
            for i, j in links:
                w = np.sum(I[tuple((P[i, 3:5] + alpha * (P[j, 3:5] - P[i, 3:5])).astype(int))] for alpha in checkpoints)
                if w < (I[tuple(P[j, 3:5])] + I[tuple(P[i, 3:5])]) / 2 * n_checkpoints * parameters['gap_bar']:
                    A[i, j] = False

            # finding the connected components (trees)
            cc_list = nx.connected_components(nx.from_numpy_matrix(A + A.T))  # list of sets

            labels = np.zeros(A.shape[0])
            for cc in cc_list:
                if len(cc) > 1:
                    labels[list(cc)] = group_id
                group_id += 1

            # save the results in a text file
            np.savetxt(os.path.join(output, fuse_filename), np.column_stack((P[:,0],labels)), fmt=['%07d', '%06d'], delimiter=',')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuses", help="path to fuse files")
    parser.add_argument("--images", help="path to CNN predictions")
    parser.add_argument("--output", help="path to output fuse files")
    parser.add_argument("--config", help="path to config file")
    parser.add_argument("--drives", help="path to drives list text-file")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    if args.verbosity >= 2:
        print(__doc__)

    run(args.fuses, args.images, args.output, args.config, args.drives, args.verbosity)


if __name__ == '__main__':
    main(sys.argv[1:])
