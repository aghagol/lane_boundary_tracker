import fcn_image_generation_and_prediction
import fcn_prediction
import os
import datetime


def run_image_generation_and_prediction(drive_id, drive_uri, model, output, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": Top-down image generation/FCN Prediction"

    if not os.path.exists(output):
        os.mkdir(output)

    fcn_image_generation_and_prediction.run(drive_id, drive_uri, model, output)


def run_fcn_prediction(drive_id, images, model, output, verbosity):
    if verbosity >= 1:
        print datetime.datetime.now(), ": FCN Prediction"

    if not os.path.exists(output):
        os.mkdir(output)

    fcn_prediction.run(drive_id, images, model, output)
