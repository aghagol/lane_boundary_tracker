#!/usr/bin/env python
""" 
This is a script for performing a sequece of postprocessing operations on the tracking resutls
"""

import sys
import os
import numpy as np
import pandas as pd
import argparse
import json
from jsmin import jsmin

import postprocessing_util


def run(seq_list, img2fuse, fuse2seq, output, graphs, config, verbosity):
    # format for output file (similar to tracking output)
    # format: frame_id, target_id, x, y, detection_id, confidence
    out_fmt = ['%05d', '%05d', '%011.5f', '%011.5f', '%05d', '%04.2f']

    with open(config) as fparam:
        param = json.loads(jsmin(fparam.read()))["postprocess"]

    flag_stitch = param['stitch_tracklets']
    flag_reduce = param['point_reduction']
    flag_fusion = param['fusion']
    flag_postprocess = flag_stitch or flag_reduce or flag_fusion

    if not os.path.exists(output):
        os.makedirs(output)

    seqs = pd.read_csv(seq_list)

    if not flag_postprocess:
        if verbosity >= 1:
            print('No post-processing requested; linking to tracker output.')
        for seq_idx, seq in seqs.iterrows():
            os.system('ln -s %s %s' % (seq.tpath, '%s/%s.txt' % (output, seq.sname)))

    #create mapping seq_name --> fuse_name --> image_name
    fuse2seq_df = pd.read_csv(fuse2seq, header=None)
    seq2fuse_dict = dict(zip(fuse2seq_df[1], fuse2seq_df[0]))
    # img2fuse_df = pd.read_csv(img2fuse, header=None)
    # fuse2img_dict = dict(zip(img2fuse_df[1], img2fuse_df[0]))

    for seq_idx, seq in seqs.iterrows():
        output_path_final = '%s/%s.txt' % (output, seq.sname)
        if os.path.exists(output_path_final): continue

        if verbosity >= 2:
            print('Working on sequence %s' % seq.sname)

        #find the corresponding image_name, fuse_name and drive_id
        drive_id = '_'.join(seq.sname.split('_')[:2])
        fuse_name = seq2fuse_dict[seq.sname]
        # image_name = fuse2img_dict[fuse_name]

        # start with the original tracking results
        trks = np.loadtxt(seq.tpath, delimiter=',')
        trks = trks[trks[:, 4] > 0, :]  # remove the guide?

        if flag_fusion:
            if verbosity >= 2:
                print('\tFusion with image based peak points clusterings')
            groups = np.loadtxt(os.path.join(graphs, fuse_name), delimiter=',')
            # groups = groups[groups[:,1]>0,:] #remove isolate points
            trks = postprocessing_util.fusion(trks, groups, param)

        if flag_reduce:
            if verbosity >= 2:
                print('\tApplying point reduction')
            trks = postprocessing_util.reducer(trks, param)

        if flag_stitch:
            if verbosity >= 2:
                print('\tApplying tracklet stitching')
            trks = postprocessing_util.stitch(trks, param)

        np.savetxt(output_path_final, trks, fmt=out_fmt, delimiter=',')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to a CSV file containing sequence-related paths")
    parser.add_argument("--img2fuse", help="path to file containing image to fuse map")
    parser.add_argument("--fuse2seq", help="path to file containing fuse to seq map")
    parser.add_argument("--output", help="output path to save tracklet fusion results")
    parser.add_argument("--graphs", help="path to image-based point clustering information")
    parser.add_argument("--config", help="configuration JSON file")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    if args.verbosity >= 2:
        print(__doc__)

    run(args.input, args.img2fuse, args.fuse2seq, args.output, args.graphs, args.config, args.verbosity)


if __name__ == '__main__':
    main(sys.argv[1:])
