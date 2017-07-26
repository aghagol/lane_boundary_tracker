import png
import os
import sys
import scipy
from scipy.misc import imsave
import argparse
from utils.cnn import predict
from utils.nn_lidar_roads.params_dir import had_6cm_mc_batchnrm


def run_prediction(drive_id, images, predictor, out_dir):
    output = os.path.join(out_dir, 'results', drive_id)

    try:
        os.makedirs(output)
    except:
        pass

    for image in images:
        out = os.path.join(output, os.path.basename(image))
        try:
            feat = scipy.misc.imread(image) / 255.
        except:
            print "Error reading " + image
            continue

        pred0 = predictor.predict(feat, label='allbnd')
        png.fromarray(pred0 * (2 ** 16 - 1), mode='L;16').save(out)


def run(drive_id, images, model, out_dir):
    # Setup model
    params = had_6cm_mc_batchnrm.params
    params.eval_model_path = model
    params.train_dir, _ = os.path.split(model)

    predictor = predict.Prediction(params, pad=32, mult=16)

    # Merge raster data into reasonable tiles and make predictions in world coordinates.
    run_prediction(drive_id, images, predictor, out_dir)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("drive_id", required=True)
    parser.add_argument('--images', help="List of top-down image paths ", required=True)
    parser.add_argument('--model', help="Location of model", required=True)
    parser.add_argument('--out-dir', help="Directory to save drive files and results", required=True)
    args = parser.parse_args()

    run(args.drive_id, args.images, args.model, args.out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
