import numpy as np
import os
from haversine import haversine


def generate_ITLLAL_and_tmap(input_path, output_path, clusters, tiny_subdrives, parameters):
    """
    Input: fuse files
    Output: itllal.txt, tmap.txt
    """
    for subdrive in clusters:

        # skip if output files exist for this subdrive
        mot_det_path = os.path.join(output_path, 'MOT', subdrive, 'det')
        itllal_file_path = os.path.join(mot_det_path, 'itllal.txt')
        tmap_file_path = os.path.join(mot_det_path, 'tmap.txt')

        if os.path.exists(itllal_file_path) and os.path.exists(tmap_file_path): continue

        # list of fuse files for this subdrive
        filelist = clusters[subdrive]

        # stack the points from all fuse files in filelist
        dets = []
        tmap = []
        for filename in filelist:
            if os.stat(os.path.join(input_path, filename)).st_size:
                # points format: id, latitude, longitude, altitude, timestamp
                dets.append(np.loadtxt(os.path.join(input_path, filename), delimiter=',').reshape(-1, 5))
                if parameters['fake_timestamp']:
                    tmap.append(np.loadtxt(os.path.join(input_path, filename) + '.tmap', delimiter=',').reshape(-1, 2))
        dets = np.vstack(dets)
        if parameters['fake_timestamp']:
            tmap = np.vstack(tmap)

        # apply recall if recall < 100%
        if parameters['recall'] < 1:
            dets = dets[np.random.rand(dets.shape[0]) < parameters['recall'], :]

        # re-arrange detections according to timestamps
        dets = dets[dets[:, 4].argsort(), :]

        # find detections that are very close to each other (and mark for deletion)
        if parameters['remove_adjacent_points'] and parameters['fake_confidence']:
            fake_confidence = np.random.rand(dets.shape[0])
            mark_for_deletion = []
            search_window_size = parameters['search_window_size']
            for i in range(dets.shape[0]):
                for j in range(max(i - search_window_size, 0), min(i + search_window_size, dets.shape[0])):
                    if haversine((dets[i, 1], dets[i, 2]), (dets[j, 1], dets[j, 2])) * 1000 < parameters['min_pairwise_dist']:
                        if fake_confidence[i] < fake_confidence[j]:  # keep the point with higher confidence
                            mark_for_deletion.append(i)
            dets = np.delete(dets, mark_for_deletion, axis=0)

        # remove subdrives/sequences with less than 2 points left after above processes
        if dets.shape[0] < 2:
            tiny_subdrives.add(subdrive)
            continue

        # generate the itllal array
        # format: index,timestamp,lat,lon,altitude,label
        itllal = np.empty((dets.shape[0], 6))
        itllal[:, 0] = dets[:, 0]  # global unique detection id
        itllal[:, 1] = dets[:, 4]  # timestamp
        itllal[:, 2:5] = dets[:, 1:4]  # lat,lon,altitude
        itllal[:, 5] = -1  # no GT labels

        # write itllal array to file
        if not os.path.exists(mot_det_path):
            os.makedirs(mot_det_path)

        # format: index,timestamp,lat,lon,altitude,label
        fmt = ['%07d', '%016d', '%.10f', '%.10f', '%.10f', '%02d']
        np.savetxt(itllal_file_path, itllal, fmt=fmt, delimiter=',')

        # write tmap array to file
        if parameters['fake_timestamp']:
            np.savetxt(tmap_file_path, tmap, fmt=['%012d', '%016d'], delimiter=',')
