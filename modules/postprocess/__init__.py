import print_param
import make_seq_list
import postprocess
import os
import datetime


def run(input, pre_process_path, tracks, graphs, output, config, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Postprocessing: Stitching and Point Reduction"

    if not os.path.exists(output):
        os.mkdir(output)

    print_param.run(config, verbosity)

    seq_list = os.path.join(output, 'seqs.csv')
    post_process_output = os.path.join(output, 'tracks')
    img2fuse = os.path.join(pre_process_path, 'img2fuse.txt')
    fuse2seq = os.path.join(pre_process_path, 'fuse2seq.txt')
    
    make_seq_list.run(input, tracks, seq_list, verbosity)
    postprocess.run(seq_list, img2fuse, fuse2seq, post_process_output, graphs, config, verbosity)
