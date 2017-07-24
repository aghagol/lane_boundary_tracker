from sort import print_param
from sort import sort

import os
import datetime


def run(input, output, config, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Tracking: SORT"

    if not os.path.exists(output):
        os.mkdir(output)

    tracks_output = os.path.join(output, 'tracks')
    
    print_param.run(config, verbosity)
    sort.run(input, tracks_output, config, verbosity)
