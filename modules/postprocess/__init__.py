import print_param
import make_seq_list
import postprocess
import os
import datetime


def run(input, tracks, graphs, output, config, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Postprocessing: Stitching and Point Reduction"

    if not os.path.exists(output):
        os.mkdir(output)

    print_param.run(config, verbosity)

    seq_list_output = output + "/seqs.csv"
    post_process_output = output + "/tracks"
    make_seq_list.run(input, tracks, seq_list_output, verbosity)
    postprocess.run(seq_list_output, post_process_output, graphs, config, verbosity)
