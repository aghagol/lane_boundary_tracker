#!/usr/bin/env python
"""
This script prints out preprocessing parameters to stdout
"""

import argparse
import json
from jsmin import jsmin
import sys


def run(config, verbosity):
    # read preprocessing parameters from the configuration JSON file
    with open(config) as fparam:
        param = json.loads(jsmin(fparam.read()))["preprocess"]

    # print selected (important) preprocessing parameters
    if verbosity >= 1:
        print('...Detections will be converted to MOT format')
        if param['remove_adjacent_points']:
            print('...Detection points closer than %.2f meters will be removed' % (param['min_pairwise_dist']))
        if param['recall'] < 1:
            print('...Detection recall (simulated)= %.2f %%' % (param['recall'] * 100))

    # print all preprocessing parameters
    if verbosity >= 2:
        print("\nParamteres:")
        for param_key, param_val in param.iteritems():
            print('...%s = %s' % (param_key, param_val))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    run(args.config, args.verbosity)

if __name__ == '__main__':
    main(sys.argv[1:])
