#!/usr/bin/env python
"""
This script creates a drives list file if it does not exist already
"""

import os
import sys
import argparse


def run(images, drives, verbosity):
    if not os.path.exists(drives):
        if verbosity >= 2:
            print("Generating a drive list because it is missing")

        # Assuming the following data hierarchy:
        #
        # images/drive_id/startTimestamp_endTimestamp_pred.png
        drive_names = [i for i in os.listdir(images) if i[:2] == 'HT']
        with open(drives, 'w') as fdrivelist:
            for drive_name in drive_names:
                fdrivelist.write('%s\n' % drive_name)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="path to FCN output")
    parser.add_argument("--drives", help="path to drives list file")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    if args.verbosity >= 2:
        print(__doc__)

    run(args.images, args.drives, args.verbosity)

if __name__ == '__main__':
    main(sys.argv[1:])
