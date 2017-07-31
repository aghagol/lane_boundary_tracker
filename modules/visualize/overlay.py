#!/usr/bin/env python
""" 
This is a script for plotting detections and tracks
CTRL+C for more options
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import json
from jsmin import jsmin
from scipy import misc

# suppress matplotlib warnings
import warnings

def run(seq_list, images, fuses, img2fuse, fuse2seq, groundtruth, overlay_out, config, verbosity):
    warnings.filterwarnings("ignore")
    
    if verbosity>=2:
        print(__doc__)

    with open(config) as fparam:
        param = json.loads(jsmin(fparam.read()))["visualize"]

    #create mapping seq_name --> fuse_name --> image_name
    img2fuse_df = pd.read_csv(img2fuse, header=None)
    fuse2seq_df = pd.read_csv(fuse2seq, header=None)
    fuse2img_dict = dict(zip(img2fuse_df[1], img2fuse_df[0]))
    seq2fuse_dict = dict(zip(fuse2seq_df[1], fuse2seq_df[0]))

    seqs = pd.read_csv(seq_list)

    for seq_idx,seq in seqs.iterrows():
        fig, ax = plt.subplots(1, 1)
        ax.set_axis_off()

        #find the corresponding image and drive_id
        fuse_name = seq2fuse_dict[seq.sname]
        prediction_image = fuse2img_dict[fuse_name]
        raw_image = fuse2img_dict[fuse_name].replace('_pred.png', '_raw.png')

        drive_id = '_'.join(seq.sname.split('_')[:2])

        if verbosity>=2:
            print('Working on seq %s (image %s, part of drive %s)'%(seq.sname, image_name, drive_id))

        if param['overlay_on_topdown'] and os.path.exists(os.path.join(images, drive_id, raw_image)):
            pred_im = misc.imread(os.path.join(images, drive_id, raw_image), mode='F') / 255.
            pred_im = np.fliplr(pred_im.T) #this is required for Jim's generated images
            ax.imshow(pred_im, cmap='gray')
        elif os.path.exists(os.path.join(images, drive_id, prediction_image)):
            pred_im = misc.imread(os.path.join(images, drive_id, prediction_image), mode='F') / (2 ** 16 - 1)
            pred_im = np.fliplr(pred_im.T) #this is required for Jim's generated images
            ax.imshow(pred_im, cmap='gray', vmin=0, vmax=2)

        #load tracking results
        dets = np.loadtxt(seq.dpath, delimiter=',')
        trks = np.loadtxt(seq.tpath,delimiter=',')

        #get pixel coordinates for each detection_id
        dets = dets[dets[:,6]>0, :] #remove the guide (fake detections)
        dets_fuse = pd.read_csv(os.path.join(fuses, fuse_name), header=None)
        dets_row_dict = dict(zip(dets_fuse[0], dets_fuse[3]))
        dets_col_dict = dict(zip(dets_fuse[0], dets_fuse[4]))

        #plot detections
        if param['overlay_detections']:
            dets_rows = [dets_row_dict[det_id] for det_id in dets[:,6]]
            dets_cols = [dets_col_dict[det_id] for det_id in dets[:,6]]
            ax.plot(dets_cols, dets_rows, 'go', markerfacecolor='None', ms=10)

        #remove the predictions that are not matched with detections
        if param['hide_predictions']:
            trks = trks[trks[:, 4] > 0, :]

        for lb_number, target_id in enumerate(sorted(set(trks[:, 1]))):

            trk_active = trks[trks[:,1]==target_id, :]
            dets_ids = trk_active[:, 4].astype(int).tolist()

            #prune short lane boundaries (polylines)
            flag_too_few_points = len(dets_ids)<param['min_seq_length']

            #prune tracks that lie in a bbox of size smaller than min_bbox_size
            bbox_x = trk_active[:,2].max()-trk_active[:,2].min() #width
            bbox_y = trk_active[:,3].max()-trk_active[:,3].min() #height
            flag_too_small = max(bbox_x,bbox_y)<param['min_bbox_size']

            flag_skip_track = flag_too_few_points or flag_too_small

            if flag_skip_track: continue

            node_rows = [dets_row_dict[det_id] for det_id in dets_ids]
            node_cols = [dets_col_dict[det_id] for det_id in dets_ids]

            c = np.array([np.random.rand(),np.random.rand()/4,1])[np.random.permutation(3)]
            ax.plot(node_cols, node_rows, color=c)

        if param['overlay_save_as_image']:
            if not os.path.exists(overlay_out):
                os.makedirs(overlay_out)
            plt.savefig(os.path.join(overlay_out, image_name), ppi=300)
        else:
            plt.show()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV file containing list of sequences")
    parser.add_argument("--output", help="where to put overlay images")
    parser.add_argument("--images", help="path to CNN predictions")
    parser.add_argument("--fuses", help="path to fuse files")
    parser.add_argument("--img2fuse", help="path to file containing image to fuse map")
    parser.add_argument("--fuse2seq", help="path to file containing fuse to seq map")
    parser.add_argument("--groundtruth", help="Show ground-truth",action='store_true')
    parser.add_argument("--config", help="configuration JSON file")
    parser.add_argument("--verbosity", help="verbosity level",type=int)
    args = parser.parse_args()


    run(args.input, args.images, args.fuses, args.img2fuse, args.fuse2seq, args.groundtruth, args.output, args.config, args.verbosity)

if __name__ == '__main__':
    main(sys.argv[1:])
