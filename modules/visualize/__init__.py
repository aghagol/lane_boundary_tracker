import print_param
import make_seq_list
import disp
import overlay
import os
import datetime


def run(mot_path, images, pre_process_path, post_process_tracks_path, visualize_path, config, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Visualization"

    if not os.path.exists(visualize_path):
        os.mkdir(visualize_path)

    img2fuse = os.path.join(pre_process_path, 'img2fuse.txt')
    fuse2seq = os.path.join(pre_process_path, 'fuse2seq.txt')

    print_param.run(config, verbosity)
    make_seq_list.run(mot_path, post_process_tracks_path, visualize_path, verbosity)
    overlay.run(mot_path, images, img2fuse, fuse2seq, True, config, verbosity)
