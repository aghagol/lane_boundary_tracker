import sys
import os
import shutil
import argparse
from modules import fcn


def run(drive_id, images, model, cache, clear_cache, verbosity):
    if os.path.exists(cache) and clear_cache:
        shutil.rmtree(cache)
    if not os.path.exists(cache):
        os.mkdir(cache)

    fcn.run_fcn_prediction(drive_id, images, model, cache, verbosity)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-id", help="drive id, from drive_metadata.json[runId]")
    parser.add_argument("--images", help="list of top down images to process")
    parser.add_argument("--model", help="path to model, eg. /model/had_6cm_mc_batchnrm2_100_rate0.1/model-54970")
    parser.add_argument("--cache", help="path to cache", default="/tmp/fcn")
    parser.add_argument("--clear_cache", help="clear existing cache before running", action='store_true')
    parser.add_argument("--verbosity", help="verbosity level", type=int, default=1)
    args = parser.parse_args()

    if args.verbosity >= 2:
        print(__doc__)

    run(args.drive_id, args.images, args.model, args.cache, args.clear_cache, args.verbosity)


if __name__ == '__main__':
    main(sys.argv[1:])
