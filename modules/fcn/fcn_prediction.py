import png
import os
import sys
import scipy
from scipy.misc import imsave
import argparse
from utils.cnn import predict
from utils.nn_lidar_roads.params_dir import had_6cm_mc_batchnrm
import json


def run_prediction(drive_id, images, predictor, out_dir):
    output = os.path.join(out_dir, 'results', drive_id)
    try:
        os.makedirs(output)
    except:
        return

    meta_csv_path = os.path.join(output, "meta.csv")
    with open(meta_csv_path, 'w') as meta_csv:
        meta_csv.write('name, time_start, time_end, min_lat, min_lon, max_lat, max_lon\n')
        for image in images:
            metadata_path = image.replace('.png', '-metadata.json')
            if not os.path.exists(metadata_path):
                continue
            with open(metadata_path) as json_file:
                metadata = json.load(json_file)

            ts_min = metadata['timeRange']['min']
            ts_max = metadata['timeRange']['max']

            min_lon = min(metadata['boundRegion']['west'], metadata['boundRegion']['east'])
            max_lon = max(metadata['boundRegion']['west'], metadata['boundRegion']['east'])
            min_lat = min(metadata['boundRegion']['north'], metadata['boundRegion']['south'])
            max_lat = max(metadata['boundRegion']['north'], metadata['boundRegion']['south'])

            name = '{}_{}_pred.png'.format(ts_min, ts_max)
            out = os.path.join(output, name)

            meta_csv.write('{}, {}, {}, {}, {}, {}, {}\n'.format(name, ts_min, ts_max,
                                                                 min_lat, min_lon, max_lat, max_lon))
            try:
                # feat = scipy.misc.imread(image)[:, :, :2] / 255.
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
    parser.add_argument('--images', help="Image path ", required=True)
    parser.add_argument('--model', help="Location of model", required=True)
    parser.add_argument('--out-dir', help="Directory to save drive files and results", required=True)
    args = parser.parse_args()

    run(args.drive_id, args.images, args.model, args.out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
