import generate_point_graph
import os
import datetime


def run(fuses, images, output, config, drives, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Point Graph Inference (Image-based)"

    if not os.path.exists(output):
        os.mkdir(output)

    generate_point_graph.run(fuses, images, output, config, drives, verbosity)
