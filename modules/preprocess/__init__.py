import os
import datetime
import print_param
import generate_drive_list
import extract_points
import tag_points
import generate_MOT_data


def run(images, output, config, drives, poses, tagged, fuses, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Pre-processing"

    if not os.path.exists(output):
        os.mkdir(output)

    print_param.run(config, verbosity)
    generate_drive_list.run(images, drives, verbosity)

    if verbosity >= 1:
        print datetime.datetime.now(), ": Pre-processing: Extracting detection points"

    extract_points.run(images, fuses, config, drives, output, verbosity)

    if verbosity >= 1:
        print datetime.datetime.now(), ": Pre-processing: Tagging (timestamps)"

    tag_points.run(fuses, tagged, images, config, drives, poses, verbosity)

    if verbosity >= 1:
        print datetime.datetime.now(), ": Pre-processing: Generating MOT"

    generate_MOT_data.run(tagged, output, config, drives, poses, verbosity)
