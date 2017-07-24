import sys
import os
import shutil
import argparse
from modules import preprocess
from modules import gengraph
from modules import trackers
from modules import postprocess
from modules import fuse
from modules import visualize


def run(images, fuses, tagged, cache, drives, poses, chunks, config, should_visualize, verbosity):
    if os.path.exists(cache):
        shutil.rmtree(cache)
    os.mkdir(cache)

    pre_process_path = os.path.join(cache, 'preprocess')
    mot_path = os.path.join(pre_process_path, 'MOT')
    graph_path = os.path.join(cache, 'graphs')
    tracker_path = os.path.join(cache, 'sort')
    tracks_path = os.path.join(tracker_path, 'tracks')
    post_process_path = os.path.join(cache, 'postprocess')
    post_process_tracks_path = os.path.join(post_process_path, 'tracks')
    fusion_path = os.path.join(cache, 'fusion')
    visualize_path = os.path.join(cache, 'visualize')

    preprocess.run(images, pre_process_path, config, drives, poses, tagged, fuses, verbosity)
    gengraph.run(fuses, images, graph_path, config, drives, verbosity)
    trackers.run(mot_path, tracker_path, config, verbosity)
    postprocess.run(mot_path, tracks_path, graph_path, post_process_path, config, verbosity)
    fuse.run(mot_path, post_process_tracks_path, fusion_path, config, chunks, verbosity)

    if should_visualize == 1:
        visualize.run(mot_path, images, pre_process_path, post_process_tracks_path, visualize_path, config, verbosity)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="path to fcn predictions")
    parser.add_argument("--fuses",  help="path to fuse files")
    parser.add_argument("--tagged", help="path to tagged fuse files")
    parser.add_argument("--cache",  help="path to cache", default="/tmp/surface_street_line_connector")
    parser.add_argument("--drives", help="path to drives list file")
    parser.add_argument("--poses",  help="path to drive pose CSV files")
    parser.add_argument("--chunks", help="path to chunk metadata")
    parser.add_argument("--config", help="path to config file", required=True)
    parser.add_argument("--visualize", help="visualize 1 = yes, 0 = no (default)", type=int, default=0)
    parser.add_argument("--verbosity", help="verbosity level", type=int, default=1)
    args = parser.parse_args()

    if args.verbosity >= 2:
        print(__doc__)

    run(args.images, args.fuses, args.tagged, args.cache, args.drives, args.poses, args.chunks, args.config,
        args.visualize, args.verbosity)


if __name__ == '__main__':
    main(sys.argv[1:])
