#!/usr/bin/env python
"""
This script prints out parameters that were used for JSON to MOT conversion
"""
import sys
import argparse
import json
from jsmin import jsmin


def run(config, verbosity):
    with open(config) as fparam:
        param = json.loads(jsmin(fparam.read()))["sort"]

    if verbosity >= 1:
        print(
        '...Detections to tracks matching distance threshold - tight = %.2f meters' % (param['d2t_dist_threshold_tight']))
        print(
        '...Detections to tracks matching distance threshold - loose = %.2f meters' % (param['d2t_dist_threshold_loose']))
        # print('...Max age (after last update) = %d frames'%(param['max_age_after_last_update']))
        print('...Max age (after last update) = %d meters' % (param['max_mov_after_last_update']))

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
