#!/usr/bin/env python
""" 
This is a script for merging fuse files from drive segments for each drive
lane marking outputs from previous stage must be grouped according to drive IDs
"""

import sys
import os
from shutil import copyfile
import argparse
import json
from jsmin import jsmin


def run(input, output, config, verbosity):
    with open(config) as fparam:
        param = json.loads(jsmin(fparam.read()))["output"]

    if not param['merge_subdrives']:
        if verbosity >= 1 and param['enable']:
            print('...Subdrive merging is disabled')
        return

    if not os.path.exists(output):
        os.makedirs(output)

    # get a list of sequence names
    fuse_dir_list = [i for i in os.listdir(input) if os.path.isdir(input + '/' + i)]

    # compute a mapping between sequence names and drive IDs
    # note: sequence names have the structure: driveID_part_number
    fuse_to_drive_map = {fuse_dir: '_'.join(fuse_dir.split('_')[:2]) for fuse_dir in fuse_dir_list}

    # for each drive, create a new folder and put its lane markings there
    drive_list = sorted(set(fuse_to_drive_map.values()))
    for drive in drive_list:
        if not os.path.exists(output + '/' + drive):
            os.makedirs(output + '/' + drive)

    # copy fuse files into respective drive_id folders
    # note: fuse files must also be renamed because two sequences within the same drive can have
    # fuse files with same names while corresponding to different lane markings
    line_counter = {drive: 0 for drive in drive_list}
    for fuse_dir in fuse_dir_list:
        fuse_files = [i for i in os.listdir(input + '/' + fuse_dir) if i.endswith('_laneMarking.fuse')]
        for fuse_file in fuse_files:
            output_file = output + '/' + fuse_to_drive_map[fuse_dir] + '/%d_laneMarking.fuse' % (
                line_counter[fuse_to_drive_map[fuse_dir]])
            if not os.path.exists(output_file):
                copyfile(input + '/' + fuse_dir + '/' + fuse_file, output_file)
            line_counter[fuse_to_drive_map[fuse_dir]] += 1

    # print some useful info
    for drive in line_counter:
        if verbosity >= 2:
            print('found %d boundaries in drive %s' % (line_counter[drive], drive))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to fuse output folder")
    parser.add_argument("--output", help="path to merged output folder")
    parser.add_argument("--config", help="configuration JSON file")
    parser.add_argument("--verbosity", help="verbosity level", type=int)
    args = parser.parse_args()

    if args.verbosity >= 2:
        print(__doc__)

    run(args.input, args.output, args.config, args.verbosity)


if __name__ == '__main__':
    main(sys.argv[1:])
