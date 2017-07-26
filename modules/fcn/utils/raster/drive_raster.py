from scipy.interpolate import splrep, splev, splprep

import numpy as np
import png
import os
import time

from pypcl.processing import transform, scene_processing, util, mesh_processing
from pypcl.ama import drivedata


def do_it(drive, res=0.05):
    # Setup cache directories
    out_dir = os.path.join(drive.base_dir, drive.drive_dir, drive.drive_name, 'raster_dat')
    print out_dir

    im_dir = os.path.join(out_dir, 'images')
    try:
        os.makedirs(im_dir)
    except:
        pass

    tile_dir = os.path.join(out_dir, 'tiles')
    try:
        os.makedirs(tile_dir)
    except:
        pass

    gp_dir = os.path.join(out_dir, 'gp')
    try:
        os.makedirs(gp_dir)
    except:
        pass

    try:
        # raise Exception("Not using caching for now")
        tile_dict = np.load(os.path.join(out_dir, 'tile_blocks.npy'))[
            ()]  # , dict(tile_time_range=np.vstack(tile_time_range), tile_pose=tile_pose, tile_bboxes=np.vstack(tile_bboxes)))
        tile_pose = tile_dict['tile_pose']
        tile_time_range = [tile_dict['tile_time_range']]
        tile_bboxes = [tile_dict['tile_bboxes']]

        tile_ind = len(tile_pose)

        print "Some computation is already cached, assuming all is complete"
        return
    except:
        pass

    try:
        raise Exception("Not using caching for now")
        gp_dict = np.load(os.path.join(out_dir, 'gp_blocks.npy'))[
            ()]  # , dict(gp_pose=gp_pose, gp_pt_idx=np.vstack(gp_pt_idx))) # We
        lidar_splits = gp_dict['gp_pt_idx']
        gp_pose = gp_dict['gp_pose']
        print "Loading Cached Drive Splits"
    except:
        ### Segment drive by pose #############################
        # This makes sure there are no overlapping roads (e.g. overpasses) or loops in the drive
        time_splits = segment_pose(drive.pose[:, 1:], drive.pose[:, 0])

        ### Find indexes into lidar data file based on timestamps ################
        # This subdivides each split so that no lidar block has more than 25M points
        lidar_splits = np.array(segment_drive(drive, time_splits))
        gp_pose = [None for i in xrange(len(lidar_splits))]

        np.save(os.path.join(out_dir, 'gp_blocks.npy'),
                dict(gp_pose=gp_pose, gp_pt_idx=lidar_splits))  # np.vstack(gp_pt_idx))) # We

    # No tiles computed yet.
    tile_ind = 0
    tile_pose = []
    tile_time_range = []
    tile_bboxes = []

    for gp_ind in xrange(len(lidar_splits)):
        start = lidar_splits[gp_ind][0]
        end = lidar_splits[gp_ind][1]

        print "%d%% (%dM/%dM)" % ((100 * start) / drive.num_lidar_pts(), start / 1e6, drive.num_lidar_pts() / 1e6)

        if (gp_pose[gp_ind] is not None):
            print "\tAlready computed"
            continue

        ######################### Load lidar data ##########################################
        data, labels, models = scene_processing.setup_drive(drive, start, end - start, cleanup_intens=True)
        coord_map = models['coord_map']

        ######################## Ground Plane Extraction #####################################
        print "\tExtracting GP"
        t = time.time()

        gp_dat = extract_groundplane(data, labels)
        gp_pose[gp_ind] = transform.xyz2lla(labels['block_pos'][0], coord_map)

        print "\t\tSaving GP"
        np.save(os.path.join(gp_dir, '%d_gp.npy' % (gp_ind)), [transform.xyz2lla(gp_dat[0], coord_map), gp_dat[1]])

        print "\t\tDone (%d)" % (time.time() - t)

        ######################## Rasterize #####################################
        print "\tRasterizing"
        t = time.time()
        tiles, bbox, blocks = raster_tiles(data, labels, gp_dat, res)
        print "\t\tDone (%d)" % (time.time() - t)

        ###################### Save Rasters ############################
        print "\t\tSaving"
        t = time.time()

        block_pose = labels['block_pos'][0]
        for z in xrange(len(tiles)):
            # Update block info
            tile_time_range.append(labels['block_times'][np.array(blocks[z])])
            tile_pose.append(transform.xyz2lla(block_pose[blocks[z][0]:blocks[z][1] + 1], coord_map))
            tile_bboxes.append(transform.xyz2lla(bbox[None, z].reshape(2, 2), coord_map).flatten())

            intens = tiles[z][0] / tiles[z][2]
            intens[np.isnan(intens)] = 0

            png.fromarray(np.clip(intens / 50, 0, 1) * 255., mode='L;8').save(
                os.path.join(im_dir, '%d_intens.png' % (tile_ind)))
            np.save(os.path.join(tile_dir, '%d_tiles.npy' % (tile_ind)), util.compress_maps(tiles[z]))

            tile_ind += 1

        np.save(os.path.join(out_dir, 'gp_blocks.npy'), dict(gp_pose=gp_pose, gp_pt_idx=lidar_splits))  # We
        np.save(os.path.join(out_dir, 'tile_blocks.npy'),
                dict(tile_time_range=np.vstack(tile_time_range), tile_pose=tile_pose,
                     tile_bboxes=np.vstack(tile_bboxes)))
        print "\t\tDone (%d)" % (time.time() - t)


def resample_traj(pts, spacing=1., err=5., return_ts=False, k0=3, bnd_weight=100):
    weights = np.ones(len(pts))
    weights[0] = bnd_weight
    weights[-1] = bnd_weight
    if (len(pts) < 3):
        model, us = splprep(pts.T, k=1, task=0, s=err ** 2 * len(pts),
                            w=weights)  # If we only have 2 points (a line!), there's no reason to use cubic interpolation
    else:
        model, us = splprep(pts.T, k=np.minimum(len(pts) - 1, k0), task=0, s=err ** 2 * len(pts), w=weights)

    # TODO: the spacing between pts_pred isn't necessarily <spacing> meters apart.  Need to adjust that here
    lens = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, 1))
    t = np.linspace(0, 1, np.sum(lens) / spacing)

    pts_pred = splev(t, model)

    if (return_ts):
        return np.c_[pts_pred].T, us
    else:
        return np.c_[pts_pred].T


def segment_pose(pose_lla, timestamp):
    # Project pose into (approximate)metric coordinate frame.
    bbox_lla = np.r_[np.min(pose_lla, 0), np.max(pose_lla, 0)]
    coord_map = transform.coordinate_map(np.r_[bbox_lla[None], bbox_lla[None]])
    pose = transform.lla2xyz(pose_lla, coord_map)

    cumdist = np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(pose, 1, 0) ** 2, 1))])

    sub_pose_inds = scene_processing.subsample_pose_greedy(pose, 5)
    sub_pose = pose[sub_pose_inds]
    sub_time = timestamp[sub_pose_inds]
    sub_cumdist = cumdist[sub_pose_inds]

    neigh_inds = util.nearest_neighbors0(sub_pose, K=1000, r=50. ** 2)

    i = 1
    cur_block = [0]
    all_raster_blocks = []

    while i < len(neigh_inds):
        split = False

        k_neigh = neigh_inds[i][(neigh_inds[i] < i) * (neigh_inds[i] >= cur_block[0])]
        travel_ds = np.abs(sub_cumdist[i] - sub_cumdist[k_neigh])

        # Conflicts: any neighbors within radius 50m, that have a travel distance greater than 70m indicate a loop of some sort.
        if ((travel_ds > 70).any()):
            split = True

        if (split == True or i == len(neigh_inds) - 1):
            # Store last block
            all_raster_blocks.append(cur_block)

            # Create new block
            cur_block = [i]

            # Launch thread to rasterize the accumulated points.
        else:
            cur_block.append(i)

        i = i + 1

    # max_pose_dist=10e10
    return np.array([timestamp[0]] + [sub_time[b[0]] for b in all_raster_blocks[1:]] + [timestamp[-1]])


def segment_drive(drive, time_splits):
    # First, split drive by timestamp:
    point_splits = [0]
    for ts in time_splits[1:-1]:
        DriveFile = open(drive.download_lidar(), 'rb')
        point_splits.append(drivedata.BinarySearchSubdrive(DriveFile, drive.num_lidar_pts(), ts))
        DriveFile.close()

    point_splits.append(drive.num_lidar_pts())

    # Next, split any blocks that are too big (more than 25M points)
    lidar_blocks = []
    for i in xrange(len(point_splits) - 1):
        N = point_splits[i + 1] - point_splits[i]

        Nblocks = np.maximum(1, int(np.round(N / 25e6)))
        pt_blocks0 = np.floor(np.linspace(0, N + 1, Nblocks + 1)).astype(int)
        pt_blocks = pt_blocks0 + point_splits[i]

        for j in xrange(len(pt_blocks) - 1):
            start = pt_blocks[j]
            end = pt_blocks[j + 1]

            # Pad the start and end.
            start = np.maximum(start - 1e6, point_splits[i])
            end = np.minimum(end + 1e6, point_splits[i + 1])

            lidar_blocks.append((int(start), int(end)))

    return lidar_blocks


def extract_groundplane(data, labels):
    X0 = labels['block_pos']  # transform.interp_pose_quat(drive, block_times, models['coord_map'])

    # Remove points that correspond to collection car or things overhead -- TODO: is this ok on lombard st?
    # There are other ways to do this, e.g, don't look forward too far
    # Note that things will be a little noisy here, but we just want to remove the bulk of the points
    height = np.concatenate(
        [data['xyz'][np.concatenate(bb), 2] - X0[0][i, 2] for i, bb in enumerate(labels['beam_blocks'])])

    bad = np.ones(len(data['xyz']), dtype=bool)
    block_inds = np.concatenate([np.zeros(0, dtype=int)] + [np.concatenate(bb) for bb in labels['beam_blocks']])
    bad[block_inds] = height > 3.
    bad[~labels['not_car']] = True

    xyz = data['xyz'][~bad]
    xyzS = scene_processing.downsample(xyz, 0.25)
    xyz_gp, Ngp, _ = mesh_processing.extract_groundplane(xyzS)

    return [xyz_gp, Ngp]


def precompute_inds(pts, res, bbox=None):
    if (bbox is None):
        bbox = np.r_[np.min(pts[:, :2], 0), np.max(pts[:, :2], 0)]

    ok = ((pts[:, :2] >= bbox[:2]) * (pts[:, :2] <= bbox[2:])).all(1)
    pts_ok = pts[ok]

    inds = ((pts_ok[:, :2] - bbox[:2]) / res).astype(int)
    shape = np.ceil((bbox[2:] - bbox[:2]) / res).astype(int)
    lin_ind = np.ravel_multi_index(inds.T, shape)

    return lin_ind, ok, shape, bbox


def raster_tiles(data, labels, gp_dat, res=0.05):
    # Split tiles so the images aren't too big
    blocks, bboxes = split_tiles(data, labels)

    block_pose = labels['block_pos'][0]
    all_maps = []
    for b_i in xrange(len(blocks)):
        # print "\t%d"%(b_i)
        # Get the point inds:
        # Remove collection car:
        inds = []
        for b in blocks[b_i]:
            indT = np.concatenate(labels['beam_blocks'][b])

            # Check radius
            k_rad = np.sqrt(np.sum((data['xyz'][indT] - block_pose[b]) ** 2, 1)) < 40.

            inds.append(indT[labels['not_car'][indT] * k_rad])
        inds = np.concatenate(inds)

        # Crop to bbox    
        pts = data['xyz'][inds]
        # TODO: cleanup intensity:  some beams are badly calibrated.  Maybe we do this in setup_drive?
        maps, _ = raster_tile(pts, data['intens'][inds], gp_dat, bbox=bboxes[b_i], res=res)
        all_maps.append(maps)

    return all_maps, bboxes, [(b[0], b[-1]) for b in blocks]  # Keep track of start and end blocks


def split_tiles(data, labels):
    # block_times = np.array([np.mean(data['time'][np.concatenate(block)]) for block in labels['beam_blocks']])
    block_pose = labels['block_pos'][0]
    block_bbox = np.c_[block_pose - [40, 40, 0], block_pose + [40, 40, 0]]  # Fixed bounding box.

    i = 1
    cur_bbox = np.r_[block_bbox[0, :2], block_bbox[0, 3:5]]
    cur_block = [0]
    all_raster_blocks = []
    all_raster_bbox = []

    max_size = 25 * 1024 ** 2  # 50mb per tile
    # max_size=10e10
    res = 0.05  # 0.05m/pixel
    max_area = (max_size / 4) * (res * res)  # max_size/4bytes per float)
    # This should give about 80mx80m for the current params
    # max_pose_dist = 80 #np.sqrt(max_area) # travel about 80m.  Not sure what we want this to be in reality...
    max_pose_dist = 80

    cumdist = np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(block_pose, 1, 0) ** 2, 1))])

    while i < len(block_bbox):
        # pull next lidar block off the queue.

        # Walk along until 1) cumulative bbox gets too big
        #                  2) travel too far long pose (this might be subsumed by 1)
        split = False
        # Conflict 1: cumulative bbox size (50mb?)

        # This can be done in LLA
        new_bbox = np.r_[np.minimum(cur_bbox[:2], block_bbox[i][:2]), np.maximum(cur_bbox[2:], block_bbox[i][3:5])]
        new_area = np.prod(new_bbox[2:] - new_bbox[:2])

        if (new_area > max_area):
            split = True

        # Conflict 2: travel too far along pose
        if ((cumdist[i] - cumdist[cur_block[0]]) > max_pose_dist):
            split = True

        #######################################################
        if (split == True or i == len(block_bbox) - 1):
            # print "new box at %d"%(i)
            # Store last block
            all_raster_blocks.append(cur_block)
            all_raster_bbox.append(cur_bbox)

            # Create new block
            cur_bbox = np.r_[block_bbox[i, :2], block_bbox[i, 3:5]]
            cur_block = [i]

            # Launch thread to rasterize the accumulated points.
        else:
            cur_block.append(i)
            cur_bbox = new_bbox

        i += 1

    return all_raster_blocks, np.vstack(all_raster_bbox)


def raster_tile(pts, intens0, gp_dat, bbox=None, res=0.05):
    lin_inds, ok_pts, shape, bbox = precompute_inds(pts, res, bbox=bbox)  # bbox)
    intens = intens0[ok_pts]
    height = pts[ok_pts, 2]

    # We need to be careful about what points we use for the groundplane!
    # THat's beyond the scope of this function though

    nrm_sign = np.sign(np.dot(gp_dat[1], [0, 0, 1]))
    nrm_up = gp_dat[1] * nrm_sign[:, None]
    # We may want to precompute this.
    dvert, dhoriz, neigh_gp = mesh_processing.dist_to_plane_points_signed(pts[ok_pts], gp_dat[0], nrm_up, euc_th=9.)

    # Ground surface layer:
    k = (dvert > -0.3) * (dvert < 0.3) * (dhoriz < 1)
    gp_cnts = rasterize_points(lin_inds[k], shape, mode='density').astype(np.float32)
    gp_intens = rasterize_points(lin_inds[k], shape, intens[k], mode='accum').astype(np.float32)
    gp_height = rasterize_points(lin_inds[k], shape, height[k], mode='accum').astype(np.float32)

    # Now do upper layers:

    height_maps = []
    hsize = 1.
    for i in xrange(10):
        k = (dvert > (hsize * i + 0.3)) * (dvert < (hsize * (i + 1) + 0.1)) * (dhoriz < 9)
        height_maps.append(rasterize_points(lin_inds[k], shape, mode='density').astype(np.float32))

    return [gp_intens, gp_height, gp_cnts] + height_maps, bbox


def rasterize_points(lin_inds, shape, values=None, mode='mean'):
    if (mode == 'mean'):
        cnts = np.bincount(lin_inds)
        accum = np.zeros(shape) + np.nan
        accum.flat[:len(cnts)] = np.bincount(lin_inds, values) / cnts
    elif (mode == 'density'):
        cnts = np.bincount(lin_inds)
        accum = np.zeros(shape)
        accum.flat[:len(cnts)] = cnts
    elif (mode == 'accum'):
        accum = np.zeros(shape)
        acc_flat = np.bincount(lin_inds, values)
        accum.flat[:len(acc_flat)] = acc_flat

    # accum[~np.isfinite(accum)] = 0

    return accum


def merge_tiles(tile_dat, inds, do_height=False, full_bbox=None, res=0.05):
    # TODO: I'm not sure things are lining up perfectly here...
    # res = 0.05
    sub_bboxes = tile_dat['tile_bboxes'][inds]

    if (full_bbox is None):
        full_bbox = np.r_[np.min(sub_bboxes[:, :2], 0), np.max(sub_bboxes[:, 2:], 0)]

    coord_map = transform.coordinate_map(np.r_[np.c_[full_bbox[None, :2, ], 0], np.c_[full_bbox[None, 2:], 0]])

    sub_bbox_xys = transform.lla2xyz(sub_bboxes.reshape(-1, 2), coord_map).reshape(-1, 4)
    bbox_xy = transform.lla2xyz(full_bbox[None].reshape(2, 2), coord_map).flatten()

    targ_shape = np.ceil((bbox_xy[2:] - bbox_xy[:2]) / res).astype(int)

    Iaccum = np.zeros(targ_shape)
    Haccum = np.zeros(targ_shape)
    Vaccum = np.zeros(targ_shape)

    sub_bbox_pix = np.c_[((sub_bbox_xys[:, :2] - bbox_xy[:2]) / res).astype(int),
                         ((sub_bbox_xys[:, 2:] - bbox_xy[:2]) / res).astype(int)]

    for i, tile_ind in enumerate(inds):
        tile_maps = util.decompress_maps(np.load(os.path.join(tile_dat['tile_dir'], '%d_tiles.npy' % (tile_ind))))
        im_inds = np.c_[np.nonzero(tile_maps[2])]  # Translate these into xy coordinates
        sub_bbox_xy = sub_bbox_xys[i]

        pix_w = (sub_bbox_xy[2:] - sub_bbox_xy[:2]) / (tile_maps[0].shape)
        pix_loc = (im_inds) * pix_w + sub_bbox_xy[:2]  # Coordinate of each pixel in xy

        # Convert to pixel in target image:
        tpix_loc = np.round((pix_loc - bbox_xy[:2]) / res).astype(int)  # I think this needs to be fixed!

        k = ((tpix_loc >= 0) * (tpix_loc < targ_shape)).all(1)
        tpix_lin_ind = np.ravel_multi_index(tpix_loc[k].T, targ_shape)
        pix_lin_ind = np.ravel_multi_index(im_inds[k].T, tile_maps[2].shape)

        Icnts = np.bincount(tpix_lin_ind, tile_maps[0].flat[pix_lin_ind])
        Hcnts = np.bincount(tpix_lin_ind, tile_maps[1].flat[pix_lin_ind])
        cnts = np.bincount(tpix_lin_ind, tile_maps[2].flat[pix_lin_ind])

        # Do loop checking here as well:  make sure height isn't totally inconsistent.
        Iaccum.flat[:len(Icnts)] += Icnts
        Haccum.flat[:len(Icnts)] += Hcnts
        Vaccum.flat[:len(Icnts)] += cnts

    return (Iaccum / Vaccum,
            Haccum / Vaccum,
            Vaccum,
            transform.xyz2lla(bbox_xy[None].reshape(2, 2), coord_map).flatten(),
            sub_bbox_pix)


def warp_vu2ll(Gparams, res, src_bbox_lla, shape):
    pix_res = (src_bbox_lla[2:] - src_bbox_lla[:2]) / shape

    res_ll = np.zeros(shape)

    todo = res > 0
    val = res[todo]
    pt_vu0 = np.c_[np.nonzero(todo)]  # Pixel indexes
    pt_vu = pt_vu0.astype(float)
    pt_vu[:, 1] = pt_vu0[:, 1] * Gparams['u_res'] - Gparams['u_win']

    pt_xy = transform.vu2xy(pt_vu[:, 0], pt_vu[:, 1], Gparams['Xt'])
    pt_ll = transform.xyz2lla(pt_xy, Gparams['coord_map'])

    pt_pix = ((pt_ll - src_bbox_lla[:2]) / pix_res).astype(int)
    k = ((pt_pix < shape) * (pt_pix >= 0)).all(1)

    pix_lin_ind = np.ravel_multi_index(pt_pix[k].T, res_ll.shape)

    val_cnts = np.bincount(pix_lin_ind, val[k])
    cnts = np.bincount(pix_lin_ind)

    val_avg = val_cnts / cnts
    val_avg[~np.isfinite(val_avg)] = 0
    res_ll.flat[:len(cnts)] = val_avg

    return res_ll


def warp_ll2vu(Gparams, Im, Hm, Vm, src_box_lla):
    # plot(Gparams['Xt'][:,0], Gparams['Xt'][:,1],'k')
    # TODO: Don't form full image...
    # TODO: Images are stored transposed.  This is because xy -> (lon,lat), but we use lat, lon.
    shape = (len(Gparams['Xt']), int(Gparams['u_win'] * 2 / Gparams['u_res']))
    Haccum = np.zeros(shape)
    Iaccum = np.zeros(shape)
    Vaccum = np.zeros(shape)
    accum = np.zeros(shape)

    # We should be able to use PIL to do this.
    inds = np.c_[np.nonzero(Vm)]  # Translate these into xy coordinates
    # inds = np.c_[np.nonzero((Im!=0) + (Hm!=0) + (Vm!=0))] # Translate these into xy coordinates
    bbox_xy = transform.lla2xyz(src_box_lla.reshape(2, 2), Gparams['coord_map']).flatten()
    pix_w = (bbox_xy[2:] - bbox_xy[:2]) / Vm.shape

    pix_loc = inds * pix_w + bbox_xy[:2]
    v, u = transform.xy2vu(pix_loc, Gparams['Xt'])

    vQ = np.round(v).astype(int)
    uQ = np.round((u + Gparams['u_win']) / Gparams['u_res']).astype(int)

    k = (v > 0) * (v < len(Gparams['Xt']) - 1) * (uQ >= 0) * (uQ < shape[1]) * (np.isfinite(u))
    vu_lin_ind = np.ravel_multi_index(np.c_[vQ[k], uQ[k]].T, shape)
    pix_lin_ind = np.ravel_multi_index(inds[k].T, Vm.shape)

    Icnts = np.bincount(vu_lin_ind, Im.flat[pix_lin_ind])
    Hcnts = np.bincount(vu_lin_ind, Hm.flat[pix_lin_ind])
    Vcnts = np.bincount(vu_lin_ind, Vm.flat[pix_lin_ind])
    cnts = np.bincount(vu_lin_ind)  # , Vm.flat[pix_lin_ind])

    # Do loop checking here as well:  make sure height isn't totally inconsistent.
    Iaccum.flat[:len(Icnts)] += Icnts
    Haccum.flat[:len(Icnts)] += Hcnts
    Vaccum.flat[:len(Icnts)] += Vcnts
    accum.flat[:len(Icnts)] += cnts

    return Iaccum / accum, Haccum / accum, Vaccum / accum
