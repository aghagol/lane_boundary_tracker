import png
import numpy as np
import os
import sys
from scipy.ndimage.filters import *
from scipy.misc import imsave
import argparse
from matplotlib import cm

from pypcl.ama import drivedata
from utils.cnn import predict
from utils.raster import drive_raster, link_raster

from utils.nn_lidar_roads.params_dir import had_6cm_mc_batchnrm
import time


def run_prediction(drive_id, drive, P, res=0.06):
    in_dir = os.path.join(drive.base_dir, drive.drive_dir, drive.drive_name, 'raster_dat')
    tile_dat = np.load(os.path.join(in_dir, 'tile_blocks.npy'))[()]
    tile_dat['tile_dir'] = os.path.join(in_dir, 'tiles')

    d = os.path.join(drive.base_dir, 'results', drive_id)

    try:
        os.makedirs(d)
    except:
        pass

    result_file = open(os.path.join(d, 'meta.csv'), 'w')
    result_file.write('name, time_start, time_end, min_lat, min_lon, max_lat, max_lon\n')

    # Render images

    N = 3
    S = np.maximum(1, N - 1)

    # Break drive into non-overlapping segments
    # time_segments = drive_raster.segment_pose(drive.pose[:,1:], drive.pose[:,0])
    # for block in xrange(len(time_segments)-1):
    #    # Find all tiles in this segment
    valid_tiles = np.arange(0, len(tile_dat['tile_time_range']))
    for i in xrange(0, len(valid_tiles), S):
        # Get time range for this series
        start = i
        end = i + N

        # Make sure everything works ok if the segment is less than 5 tiles
        if (end >= len(valid_tiles)):
            end = len(valid_tiles) - 1
            start = np.maximum(0, end - N)

        print '%d/%d (%d:%d)' % (i, len(valid_tiles) / S, start, end)

        # Merge tiles into a single image
        t = time.time()
        Im, Hm, Vm, full_bbox_lla, sub_boxes_pix = drive_raster.merge_tiles(tile_dat, np.arange(valid_tiles[start],
                                                                                                valid_tiles[end] + 1),
                                                                            res=res)
        Vm[~np.isfinite(Vm)] = 0

        ImT, HmT, VmT = link_raster.normalize_rasters(Im, Hm, Vm, w=[2.5 / res, 2.5 / res])
        print "Image is %dx%d" % (ImT.shape[0], ImT.shape[1])
        print '\tRasterizing:%f' % (time.time() - t)
        time_range = [tile_dat['tile_time_range'][valid_tiles[start], 0],
                      tile_dat['tile_time_range'][valid_tiles[end], 1]]

        t = time.time()
        feat = np.dstack([HmT, ImT])

        # Do the prediction!  This will take a little bit of time.
        pred0 = P.predict(feat, label='allbnd')

        print '\tPrediction:%f' % (time.time() - t)
        # mn = np.min(pred)
        # pred_clip = (np.clip(pred,mn,1.)-mn)/(1.-mn)
        mn = np.min(pred0)
        mx = 1

        pred = (np.clip(pred0, mn, mx) - mn) / (mx - mn)
        # pred = peak_finding.peaks_clean_pca(pred0, 0.05)*pred0

        # Warp back to lat-lon and save results for each input raster.
        vis_world(time_range, drive_id, drive, pred, feat[:, :, 1], full_bbox_lla, Hm=Hm, res=res)

        fname = '%d_%d_pred.png' % (time_range[0], time_range[1])

        result_file.write('%s, %d, %d, %.9f,%.9f,%.9f,%.9f\n' % (fname,
                                                                 time_range[0],
                                                                 time_range[1],
                                                                 full_bbox_lla[0],
                                                                 full_bbox_lla[1],
                                                                 full_bbox_lla[2],
                                                                 full_bbox_lla[3]))
        result_file.flush()


def vis_world(time_range, drive_id, drive, pred, Im, bbox_lla, Hm=None, res=0.06):
    d = os.path.join(drive.base_dir, 'results', drive_id)

    try:
        os.makedirs(d)
    except:
        pass

    t = time.time()
    mn = np.min(pred)
    mx = 1

    res_clip = (np.clip(pred, mn, mx) - mn) / (mx - mn)

    pred_im = cm.hot(res_clip)
    ImT = Im.copy()
    ImT[~np.isfinite(ImT)] = 0
    intens_im = cm.hot(ImT)

    out_im = mix_im(pred_im[:, :, :3], intens_im[:, :, np.r_[2, 0, 1]])
    print '\tVisualizing:%f' % (time.time() - t)

    t = time.time()
    imsave(os.path.join(d, '%d_%d.jpg' % (time_range[0], time_range[1])), out_im)
    print '\tSaved vis:%f' % (time.time() - t)

    t = time.time()
    imsave(os.path.join(d, '%d_%d_raw.jpg' % (time_range[0], time_range[1])), intens_im[:, :, np.r_[2, 0, 1]])
    print '\tSaved raw:%f' % (time.time() - t)

    # t=time.time()
    # np.save(os.path.join(d, '%d_%d.npy'%(time_range[0], time_range[1])), pred)
    # print '\tSaved npy:%f'%(time.time()-t)

    t = time.time()
    png.fromarray(pred * (2 ** 16 - 1), mode='L;16').save(
        os.path.join(d, '%d_%d_pred.png' % (time_range[0], time_range[1])))
    print '\tSaved pred:%f' % (time.time() - t)

    if (Hm is not None):
        # hrange = [np.min(Hm[np.isfinite(Hm)]), np.max(Hm[np.isfinite(Hm)])]
        # Hm_out = cm.jet((Hm-hrange[0])/(hrange[1]-hrange[0]))[:,:,:3]
        # png.fromarray(Hm_out*(2**16-1),mode='RGB;16').save(os.path.join(d, '%d_%d_height.png'%(time_range[0], time_range[1])))

        t = time.time()
        Hout = compress_height(Hm, res)
        png.fromarray(Hout, mode='RGB;16').save(os.path.join(d, '%d_%d_height.png' % (time_range[0], time_range[1])))
        # np.save(os.path.join(d, '%d_%d_height.png'%(time_range[0], time_range[1])), Hm)
        print '\tSaved height:%f' % (time.time() - t)


def decompress_height(Him):
    ref_height_int = Him[:, :, 0]
    height_delta_int = Him[:, :, 1]

    ref_height = (ref_height_int.astype(np.float64) - 2. ** 15)
    local_range = [-20, 20]
    height_delta = height_delta_int.astype(np.float64) / (2. ** 16 - 1) * (local_range[1] - local_range[0]) + \
                   local_range[0]

    Hf = ref_height + height_delta

    bad = ref_height_int == 0
    Hf[bad] = np.nan

    return Hf


def uniform_filter_safer(dat, win):
    # They use cumulative sums which cause floating point errors
    # This makes numerators unsafe when they're close to zero
    # This make sure that cells that are supposed to be zero are indeed zero
    # Assumptions: input is non-negative...
    Y = convolve1d(convolve1d(dat, np.ones(win), 1), np.ones(win), 0)

    return Y / float(win * win)


def uniform_filter_safe0(dat, win):
    # They use cumulative sums which cause floating point errors
    # This makes numerators unsafe when they're close to zero
    # This make sure that cells that are supposed to be zero are indeed zero
    # Assumptions: input is non-negative...
    th = np.min(dat[dat > 0]) / (win * 2.)

    dat1 = uniform_filter1d(dat, win, 0)
    dat1[dat1 <= th] = 0

    th = np.min(dat1[dat1 > 0]) / (win * 2.)
    dat2 = uniform_filter1d(dat1, win, 1)
    dat2[dat2 <= th] = 0

    return dat2


def compress_height(Hm, res):
    k = np.isfinite(Hm)
    Hk = Hm.copy()
    Hk[~k] = 0
    # ref_height0 = gaussian_filter(Hk, 2.5/res)/gaussian_filter(k.astype(float), 2.5/res) #
    ref_height0 = uniform_filter(Hk, 2.5 / res) / uniform_filter_safer(k.astype(float), int(2.5 / res))  #

    ref_height = ref_height0.astype(int)  # Average height rounded to meters

    height_delta0 = Hk - ref_height  # This is the remainder

    local_range = [-20, 20]  # For 16 bit integers, this gives .6mm resolution
    height_delta = (np.clip(height_delta0, local_range[0], local_range[1]) - local_range[0]) / (
        local_range[1] - local_range[0])

    height_delta_int = (height_delta * (2 ** 16 - 1)).astype(np.uint16)
    ref_height_int = (ref_height + 2 ** 15).astype(np.uint16)  # adding 2**15 in case there are negative altitudes
    ref_height_int[~k] = 0
    height_delta_int[~k] = 0
    final_im = np.dstack(
        [ref_height_int, height_delta_int, height_delta_int])  # Local height in both G+B channels for pretty images

    return final_im


def mix_im(im_pred, im_intens):
    a0 = np.clip(np.max(im_pred, 2) + 1e-9, 0, 1)[:, :, None]
    a1 = np.max(im_intens, 2)[:, :, None] / 4

    im_out = im_pred.astype(float) * a0 + im_intens.astype(float) * a1 * (1 - a0)
    im_out = im_out / (a0 + a1 * (1 - a0))

    return im_out


def run(drive_id, drive_uri, model, out_dir, doda=False):
    # Setup model
    params = had_6cm_mc_batchnrm.params
    params.eval_model_path = model
    params.train_dir, _ = os.path.split(model)

    predictor = predict.Prediction(params, pad=32, mult=16)

    # Setup drive
    if (doda):  # Warning: this hasn't been validated 100%.
        drive = drivedata.DriveData(drive_uri, base_dir=out_dir, align_url='http://doda.had.in.here.com',
                                    run_id=1)
    else:
        drive = drivedata.DriveData(drive_uri, base_dir=out_dir)  # TODO: specify path to raw ama files.

    # Preprocessing for top-down rasters (this includes ground segmentation)
    # This will create tiles that can be merged for arbitrary bounding boxes
    drive_raster.do_it(drive, res=0.03)  # Techinically we can do this at 0.06m resolution.

    # Merge raster data into reasonble tiles and make predictions in world coordinates.
    run_prediction(drive_id, drive, predictor)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-id")
    parser.add_argument('--drive-uri',
                        help="Drive asset uri (e.g. s3://asset.ama.here.com/db7aea32-3b38-4122-b3ae-fe8d053e1129)")
    parser.add_argument('--model', help="Location of model", required=True)
    parser.add_argument('--out-dir', help="Directory to save drive files and results", required=True)
    parser.add_argument('--doda', help="Use DODA pose correction", action="store_true", dest='doda')

    args = parser.parse_args()

    run(args.drive_id, args.drive_uri, args.model, args.out_dir, args.doda)

if __name__ == "__main__":
    main(sys.argv[1:])

