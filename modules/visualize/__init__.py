import print_param
import make_seq_list
import disp
import overlay
import os
import datetime


def run(vis_flag, mot_path, images, fuses, pre_process_path, post_process_tracks_path, visualize_path, config, verbosity):
    if vis_flag==0:
        return

    if verbosity >= 1:
        print datetime.datetime.now(), ": Visualization"

    if not os.path.exists(visualize_path):
        os.mkdir(visualize_path)

    seq_list = os.path.join(visualize_path, 'seqs.csv')
    img2fuse = os.path.join(pre_process_path, 'img2fuse.txt')
    fuse2seq = os.path.join(pre_process_path, 'fuse2seq.txt')
    overlay_out = os.path.join(visualize_path, 'overlaid')

    print_param.run(config, verbosity)
    make_seq_list.run(mot_path, post_process_tracks_path, seq_list, verbosity)

    if vis_flag==1:
        disp.run(seq_list, .01, 5, False, 30, config, verbosity)

    if vis_flag==2:
        overlay.run(seq_list, images, fuses, img2fuse, fuse2seq, False, overlay_out, config, verbosity)
