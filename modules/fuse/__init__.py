import print_param
import make_seq_list
import generate_fuse_files
import merge_sequences
import os
import datetime


def run(input, tracks, output, config, chunks, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Generating output"

    if not os.path.exists(output):
        os.mkdir(output)

    print_param.run(config, verbosity)

    fuse_output = os.path.join(output, 'seqs.csv')
    fused_output = os.path.join(output, 'fused')
    fused_merged_output = os.path.join(output, 'fused_merged')
    
    make_seq_list.run(input, tracks, fuse_output, verbosity)

    generate_fuse_files.run(fuse_output, chunks, fused_output, config, verbosity)
    merge_sequences.run(fused_output, fused_merged_output, config, verbosity)
