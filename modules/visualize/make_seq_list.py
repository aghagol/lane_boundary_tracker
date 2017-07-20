#!/usr/bin/env python
""" 
Creating a list of sequences
"""

import pandas as pd
import os
import sys
import argparse


def run(input, tracks, output, verbosity):
    sequences = sorted([i for i in os.listdir(input)])
    if os.path.exists(tracks):
        seq_set_trks = set([i[:-4] for i in os.listdir(tracks) if i.endswith('txt')])
        seq_set_dets = set(sequences)
        for seq in sequences:
            if not seq in seq_set_trks:
                seq_set_dets.remove(seq)
        sequences = sorted(list(seq_set_dets))

    seqs = {}
    seqs['name'] = sequences
    seqs['dpath'] = ['%s/%s/det/det.txt' % (input, seq) for seq in sequences]
    seqs['mpath'] = ['%s/%s/det/timestamps.txt' % (input, seq) for seq in sequences]
    seqs['gpath'] = ['%s/%s/gt/gt.txt' % (input, seq) for seq in sequences]
    seqs['tpath'] = ['%s/%s.txt' % (tracks, seq) for seq in sequences]

    seqs_df = pd.DataFrame(seqs)
    seqs_df = seqs_df[['name', 'dpath', 'tpath', 'mpath', 'gpath']]  # sort

    seqs_df.to_csv(output, index=False, header=True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to MOT dataset")
    parser.add_argument("--tracks", help="path to predicted tracks")
    parser.add_argument("--output", help="output path to save seq.csv")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    if args.verbosity >= 2:
        print(__doc__)

    run(args.input, args.tracks, args.output, args.verbosity)

if __name__ == '__main__':
    main(sys.argv[1:])
