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
        param = json.loads(jsmin(fparam.read()))["fuse"]

    if verbosity >= 1:
        if not param['enable']:
            print('...Fusion is disabled')

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
