# Given a drive and reference link, rasterize in link rectified reference frame
from scipy.ndimage.filters import *
import hashlib
import os
import glob
import png
import numpy as np
from pypcl.processing import transform, scene_processing, rmob, util

link_version = 7


def get_cached_link_list(link_dat, BASEDIR, merge_map=False, map_only=False):
    uid = hashlib.sha1(str(link_dat)).hexdigest()
    relevant_res0 = glob.glob(
        os.path.join(BASEDIR, 'data', '*', '*', 'link_raster', 'v' + str(link_version), 'raw_feat', uid,
                     '*_intens.png'))

    # Now strip out basedir and base name
    relevant_res1 = [os.path.split(fn) for fn in relevant_res0]
    relevant_res = [(a[0], a[1].replace('_intens.png', '')) for a in relevant_res1]

    return relevant_res, uid


def demo(DRIVE_ID, BASEDIR):
    drive = drivedata.load_cached_drives(DRIVE_ID, BASEDIR, strip_biggest=True)[0]
    full_bbox = np.r_[np.min(drive.pose[:, 1:3], 0), np.max(drive.pose[:, 1:3], 0)]
    coord_map = transform.coordinate_map(np.r_[np.c_[full_bbox[None, :2, ], 0], np.c_[full_bbox[None, 2:], 0]])
    rmob_raw = rmob.load_cached_rmob(np.array(full_bbox))
    rmob_links = rmob.rmob_raw2link_dat(rmob_raw, coord_map)
    final_links = rmob.get_composite_links(rmob_links, rmob_raw)

    link_raster.cache_all_link_rasters(drive, final_links, coord_map)


def cache_all_link_rasters(drive, final_links, coord_map, Gparams=None):
    pose_xyz0 = transform.lla2xyz(drive.pose[:, 1:], coord_map)
    subsamp = scene_processing.subsample_pose_greedy(pose_xyz0, 1.)  # Speeds things up
    pose_xyz = pose_xyz0[subsamp]
    timestamp = drive.pose[subsamp, 0]

    # Get timestamps 
    all_pose2link = rmob.traj2links(final_links, pose_xyz[:, :2], coarse_seg=False)
    all_time2link = dict([(k, timestamp[d]) for k, d in all_pose2link.items()])

    out_dir = os.path.join(drive.base_dir, drive.drive_dir, drive.drive_name, 'raster_dat')
    tile_dat = np.load(os.path.join(out_dir, 'tile_blocks.npy'))[()]
    tile_dat['tile_dir'] = os.path.join(out_dir, 'tiles')
    # tile_pose = np.concatenate([transform.lla2xyz(p, coord_map) for p in tile_dat['tile_pose']])
    # tile_ind = np.concatenate([i*np.ones(len(p), dtype=int) for i,p in enumerate(tile_dat['tile_pose'])])
    # dict(tile_time_range=np.vstack(tile_time_range), tile_pose=tile_pose, tile_bboxes=np.vstack(tile_bboxes)))

    base_dir = os.path.join(drive.base_dir, drive.drive_dir, drive.drive_name, 'link_raster', 'v' + str(link_version),
                            'raw_feat')

    for link_id, time_ranges in all_time2link.items():
        if (len(time_ranges) == 0):
            continue

        link_dat = final_links[link_id]
        cache_link_rasters(link_dat, tile_dat, time_ranges, coord_map, base_dir, Gparams)


def cache_link_rasters(link_dat, tile_dat, time_ranges, coord_map, base_dir, Gparams=None):
    uid = hashlib.sha1(str(link_dat)).hexdigest()

    im_dir = os.path.join(base_dir, uid)

    try:
        os.makedirs(im_dir)
    except:
        pass

    if (Gparams is None):
        Gparams = {}
        Gparams['coord_map'] = coord_map
        Gparams['v_res'] = 0.2  # Resolution along length of road
        Gparams['u_res'] = 0.05  # 5cm/pixel
        Gparams['u_win'] = 30  # +/- 30m laterally from link

    rmob_traj0 = rmob.sample_pts(link_dat['all_link_samps'], err=0.001, spacing=Gparams['v_res'], return_ts=False, k0=1)
    rmob_traj = rmob.sample_pts(rmob_traj0, err=1, spacing=Gparams['v_res'], return_ts=False)
    Gparams['Xt'] = np.c_[rmob_traj, np.zeros(len(rmob_traj))]  # I think this needs to be in 3D

    for trange in time_ranges:
        base_name = "%d_%d" % (trange[0], trange[1])
        # File tiles where time overlaps.
        # Might want to check for position conflicts again.
        Ivu0, Hvu0, Vvu0 = raster_link_range(Gparams, trange, tile_dat)

        # Save it as image for now.  It takes very little time to generate this, so we can update as needed.
        Ivu, Hvu, Vvu = normalize_rasters(Ivu0, Hvu0, Vvu0)

        # Now save as png
        png.fromarray(Ivu * 255., mode='L;8').save(os.path.join(im_dir, '%s_intens.png' % (base_name)))
        png.fromarray(Hvu * 255., mode='L;8').save(os.path.join(im_dir, '%s_height.png' % (base_name)))
        png.fromarray(Vvu * 255., mode='L;8').save(os.path.join(im_dir, '%s_vis.png' % (base_name)))


# TODO: this is duplicated in cnn.util
def normalize_rasters(I, H, C, w=[12, 50]):
    # Prepare images for saving to png.  Make sure data is in range of [0,1]

    k = np.isfinite(I)
    Ik = I.copy()
    Ik[~k] = 0
    Ik = np.clip(Ik / 100, 0, 1)

    k = np.isfinite(H)
    Hk = H.copy()
    Hk[~k] = 0

    Hlocal = H - gaussian_filter(Hk, w) / gaussian_filter(k.astype(float), w)  #
    print np.nanmin(Hlocal), np.nanmax(Hlocal)
    Hlocal = np.clip(Hlocal, -1.5, 1.5)  # This works, even for lombard st.

    Hlocal = (Hlocal + 1.5) / 3
    # Consider doing this after normalizing to [0,1].  Then a filter*0pixel can give no response.
    Hlocal[~np.isfinite(Hlocal)] = 0  # Not much we can do here...

    Ck = np.clip(np.log10(C + 1) / 4., 0, 1)

    return Ik, Hlocal, Ck


def transform_rasters(rasters, w=[12, 50]):
    # Prepare images for saving to png.  Make sure data is in range of [0,1]
    I, H, C = rasters

    k = np.isfinite(I)
    Ik = I.copy()
    Ik[~k] = 0
    Ik = np.clip(Ik / 100, 0, 1)

    k = np.isfinite(H)
    Hk = H.copy()
    Hk[~k] = 0

    Hlocal = H - gaussian_filter(Hk, w) / gaussian_filter(k.astype(float), w)  #
    print np.nanmin(Hlocal), np.nanmax(Hlocal)
    Hlocal = np.clip(Hlocal, -1.5, 1.5)  # This works, even for lombard st.

    Hlocal = (Hlocal + 1.5) / 3
    # Consider doing this after normalizing to [0,1].  Then a filter*0pixel can give no response.
    Hlocal[~np.isfinite(Hlocal)] = 0  # Not much we can do here...

    Ck = np.clip(np.log10(C + 1) / 4., 0, 1)

    return Ik, Hlocal, Ck


def raster_link_range(Gparams, trange, tile_dat):
    tile_time_range = tile_dat['tile_time_range']
    # TODO: Need to do loop checking here!
    k_tile = np.minimum(tile_time_range[:, 1], trange[1]) > np.maximum(tile_time_range[:, 0], trange[0])

    coord_map = Gparams['coord_map']

    # figure(1)
    # plot(Gparams['Xt'][:,0], Gparams['Xt'][:,1],'k')
    # TODO: Don't form full image...
    # TODO: Images are stored transposed.  This is because xy -> (lon,lat), but we use lat, lon.
    shape = (len(Gparams['Xt']), int(Gparams['u_win'] * 2 / Gparams['u_res']))
    Haccum = np.zeros(shape)
    Iaccum = np.zeros(shape)
    accum = np.zeros(shape)

    for tile_ind in np.nonzero(k_tile)[0]:
        tile_maps = np.load(os.path.join(tile_dat['tile_dir'], '%d_tiles.npy' % (tile_ind)))

        # Rasterize GP info
        gp_maps = util.decompress_maps(tile_maps[:3])  # intensity, height, counts
        inds = np.c_[np.nonzero(gp_maps[2])]  # Translate these into xy coordinates

        bbox_xy = transform.lla2xyz(tile_dat['tile_bboxes'][None, tile_ind].reshape(2, 2), coord_map).flatten()
        pix_w = (bbox_xy[2:] - bbox_xy[:2]) / gp_maps[0].shape

        pix_loc = inds * pix_w + bbox_xy[:2]
        v, u = transform.xy2vu(pix_loc, Gparams['Xt'])

        # Now rasterize!
        vQ = np.round(v).astype(int)
        uQ = np.round((u + Gparams['u_win']) / Gparams['u_res']).astype(int)

        k = (v > 0) * (v < len(Gparams['Xt']) - 1) * (uQ >= 0) * (uQ < shape[1]) * (np.isfinite(u))
        vu_lin_ind = np.ravel_multi_index(np.c_[vQ[k], uQ[k]].T, shape)
        pix_lin_ind = np.ravel_multi_index(inds[k].T, gp_maps[2].shape)

        Icnts = np.bincount(vu_lin_ind, gp_maps[0].flat[pix_lin_ind])
        Hcnts = np.bincount(vu_lin_ind, gp_maps[1].flat[pix_lin_ind])
        cnts = np.bincount(vu_lin_ind, gp_maps[2].flat[pix_lin_ind])

        # Do loop checking here as well:  make sure height isn't totally inconsistent.
        Iaccum.flat[:len(Icnts)] += Icnts
        Haccum.flat[:len(Icnts)] += Hcnts
        accum.flat[:len(Icnts)] += cnts

    return Iaccum / accum, Haccum / accum, accum
