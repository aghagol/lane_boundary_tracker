import print_param
import make_seq_list
import disp
import os
import datetime


def run(input, tracks, output, config, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Visualization"

    if not os.path.exists(output):
        os.mkdir(output)

    print_param.run(config, verbosity)
    make_seq_list.run(input, tracks, output, verbosity)
    disp.run(input, config, verbosity)
