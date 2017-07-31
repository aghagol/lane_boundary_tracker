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

# suppress matplotlib warnings
import warnings


def run(seq_list, delay, margin, groundtruth, window_size, config, verbosity):
    warnings.filterwarnings("ignore")
    w = float(window_size)
    delay = delay

    if verbosity >= 2:
        print(__doc__)

    with open(config) as fparam:
        param = json.loads(jsmin(fparam.read()))["visualize"]

    # over-ride paramaters from config file
    if 'delay' in param: delay = param['delay']
    if 'window_size' in param: w = param['window_size']

    seqs = pd.read_csv(seq_list)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    colors = np.random.rand(711, 3)  # create random colors for tracks
    frame_buffer_size = 100

    seq_idx = 0
    while seq_idx < seqs.shape[0]:
        seq = seqs.iloc[seq_idx]

        if verbosity >= 2:
            print('Working on sequence %s' % seq.sname)

        dets = np.loadtxt(seq.dpath, delimiter=',')
        dets = dets[dets[:, 6] > 0, :]  # remove the guide (fake points)
        frame_timestamps = dict(zip(dets[:, 0], dets[:, 7]))

        if param['show_tracks']:
            if groundtruth:
                if not os.path.exists(seq.gpath): exit("GT data not there!")
                trks = np.loadtxt(seq.gpath, delimiter=',')
            else:
                if not os.path.exists(seq.tpath): exit("\nNo tracks file was found!\n")
                trks = np.loadtxt(seq.tpath, delimiter=',')
                if param['hide_predictions']:
                    trks = trks[trks[:, 4] > 0, :]  # remove the predictions with no matches

        frames = sorted(frame_timestamps)
        frame_idx = 0
        while frame_idx < len(frames):
            try:
                frame = frames[frame_idx]
                timestamp = frame_timestamps[frame]

                dets_cur = dets[dets[:, 0] == frame, :]
                ax.cla()
                if param['centered']:
                    x_center = np.floor(np.median(dets_cur[:, 3]) / w) * w  # column (sub-pixel)
                    y_center = np.floor(np.median(dets_cur[:, 2]) / w) * w  # row (sub-pixel)
                    ax.set_xlim([-margin, margin + w])
                    ax.set_ylim([-margin, margin + w])
                    ax.add_patch(patches.Rectangle((0, 0), w, w, fill=False))
                else:
                    xlim_low = np.floor(np.median(dets_cur[:, 3]) / w) * w
                    ylim_low = np.floor(np.median(dets_cur[:, 2]) / w) * w
                    ax.set_xlim([xlim_low - margin - w / 2, xlim_low + margin + w / 2])
                    ax.set_ylim([ylim_low - margin - w / 2, ylim_low + margin + w / 2])
                    x_center = 0
                    y_center = 0

                # plot detections
                for plot_frame_idx in range(max(frame_idx - frame_buffer_size, 0), frame_idx):
                    ax.plot(dets[dets[:, 0] == frames[plot_frame_idx], 3] - x_center,
                            dets[dets[:, 0] == frames[plot_frame_idx], 2] - y_center, 'o', color='k')

                # plot tracks
                start_frame = frames[max(frame_idx - frame_buffer_size, 0)]
                if param['show_tracks']:
                    trks_active_set = set(trks[np.logical_and(trks[:, 0] < frame, trks[:, 0] >= start_frame), 1])
                    for trk_curr_id in trks_active_set:
                        trk_curr_tail = trks[trks[:, 1] == trk_curr_id, :]
                        if trk_curr_tail.shape[0] < param['min_seq_length']: continue
                        trk_curr_tail = trk_curr_tail[trk_curr_tail[:, 0] < frame, :]
                        trk_curr_tail = trk_curr_tail[trk_curr_tail[:, 0] >= start_frame, :]
                        ax.plot(trk_curr_tail[:, 3] - x_center, trk_curr_tail[:, 2] - y_center,
                                color=colors[int(trk_curr_id % 711), :])

                if param['real_time']:
                    delay = (frame_timestamps[frames[min(frame_idx + 1, len(frames) - 1)]] - timestamp) * 1e-6 * param[
                        'delay_multiplier']
                ax.set_title(
                    'frame number %05d/%05d, time=%.6f (+%.2f)' % (frame_idx + 1, len(frames), timestamp * 1e-6, delay))
                plt.pause(min(max(delay, 0), param['max_delay']) + .001)  # zero delay results in a halt!
                frame_idx += 1

            except KeyboardInterrupt:
                print('')
                print('=================================================')
                print('...Enter "j"[int] to jump to frame')
                print('...Enter "w"[float] to adjust window width')
                print('...Enter "d"[float] to adjust delay (disabled in realtime mode)')
                print('...Enter "q" to quit')
                print('...Enter "s" to skip to next sequence')
                print('...Enter "b" to go back one sequence')
                inp = raw_input("...or just press [ENTER] to resume: ")
                print('=================================================')
                if len(inp.strip()):
                    if inp[0] == 'j': frame_idx = int(inp[1:]) - 1
                    if inp[0] == 'w': w = float(inp[1:])
                    if inp[0] == 'd': delay = float(inp[1:])
                    if inp == 'q': exit('')
                    if inp == 's': break
                    if inp == 'b': seq_idx -= 2; break
        seq_idx += 1


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="a CSV file with 5 columns: \
      name (sequence name), dpath (path to detections), tpath (path to tracking results), \
      mpath (path to frame timestamps) and gpath (path to groundtruth - optional).")
    parser.add_argument("--delay", type=float, default=.01, help="delay in seconds for each frame.")
    parser.add_argument("--margin", type=int, default=5, help="add this many pixels to plot margins.")
    parser.add_argument("--groundtruth", action='store_true', help="Show ground-truth.")
    parser.add_argument("--window-size", type=float, default=100., help="Display window size.")
    parser.add_argument("--config", help="configuration JSON file")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    run(args.input, args.delay, args.margin, args.groundtruth, args.window_size, args.config, args.verbosity)

if __name__ == '__main__':
    main(sys.argv[1:])
