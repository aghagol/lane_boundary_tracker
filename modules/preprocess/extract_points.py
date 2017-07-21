#!/usr/bin/env python
"""
This script extracts detection points (in the fuse format) from CNN predictions
"""

import pandas as pd
import numpy as np
from scipy import misc
from scipy.ndimage.filters import maximum_filter
import os
import argparse
import json
from jsmin import jsmin

import peak_finding

parser = argparse.ArgumentParser()
parser.add_argument("--images",     help="path to CNN predictions")
parser.add_argument("--fuses",      help="path to output fuse files")
parser.add_argument("--poses",      help="path to pose CSV files")
parser.add_argument("--config",     help="path to config file")
parser.add_argument("--drives",     help="path to drives list text-file")
parser.add_argument("--output",     help="output path to MOT dataset")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

if args.verbosity>=2:
  print(__doc__)

if not os.path.exists(args.fuses):
  os.makedirs(args.fuses)

#read preprocessing parameters from the configuration JSON file
with open(args.config) as fparam:
  parameters = json.loads(jsmin(fparam.read()))["preprocess"]

#get names of drives to be processed
drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

#format of output .fuse files. 
fuse_fmt = ['%07d','%.16f','%.16f','%05d','%05d'] #format: global_peak_id, latitude, longitude

image_to_fuse_map = os.path.join(args.output,'img2fuse.txt')
f_i2f = open(image_to_fuse_map,'w')

for drive in drive_list:
  if args.verbosity>=2:
    print('Working on drive %s'%drive)

  #global counter for peaks (global within each drive)
  counter = 1

  #read metadata (in CSV format) for images of the current drive
  #meta column labels: name, time_start, time_end, min_lat, min_lon, max_lat, max_lon
  meta = pd.read_csv(os.path.join(args.images,drive,'meta.csv'), skipinitialspace=True)

  for n,res in meta.iterrows():

    #generate an image tag (must uniquely identify each image) to include in the .fuse file name
    image_tag = '%d'%(res['time_start'])
    fuse_filename = '%s_%s.png.fuse'%(drive,image_tag)

    #skip the .fuse file generation if the image does not exist (of course)
    if not os.path.exists(os.path.join(args.images,drive,res['name'])): continue

    #write image to fuse filenames mapping to file
    f_i2f.write('%s,%s\n'%(res['name'],fuse_filename))

    #skip the .fuse file generation if the .fuse file exists already
    if os.path.exists(os.path.join(args.fuses,fuse_filename)): continue

    if args.verbosity>=2:
      print('\tworking on %s'%(res['name']))

    #read the probability image (FCN output)
    pred_im = misc.imread(os.path.join(args.images,drive,res['name']))/(2.**16-1)

    if args.verbosity>=2:
      print('\t\timage size: %d x %d'%(pred_im.shape[0],pred_im.shape[1]))

    #find peaks using Ian's "peak_finding" algorithm and discard non-peak detections
    if parameters['ians_peak_filter']:
      peaks = peak_finding.peaks_clean(pred_im, 0.3, input_as_mag=True)
      pred_im = pred_im*peaks

    #discard detections that are not local maxima within a "maxpool_size" window
    pred_im[pred_im!=maximum_filter(pred_im,size=parameters['maxpool_size'])]=0

    #get lat-lon coordiantes of detection points
    bbox = np.r_[res['min_lat'],res['min_lon'],res['max_lat'],res['max_lon']]

    #extract row and column pixel indices
    #but only retain high confidence detection points
    loc_pix = np.argwhere(pred_im>parameters['global_confidence_thresh'])

    #scale to [0,1] in each direction
    loc_im = loc_pix.astype(float)/pred_im.shape

    #compute final lat-lon coordinates
    lat_lon = loc_im[:,::-1] * (bbox[2:]-bbox[:2]) + bbox[:2]

    #assign peaks with unique id's
    lat_lon = np.hstack((np.arange(lat_lon.shape[0]).reshape(-1,1)+counter,lat_lon,loc_pix))
    counter += lat_lon.shape[0]

    if args.verbosity>=2:
      print('\t\trecorded %d detection points from %s'%(lat_lon.shape[0],res['name']))

    #save detection points in .fuse files
    np.savetxt(os.path.join(args.fuses,fuse_filename),lat_lon,fmt=fuse_fmt,delimiter=',')

f_i2f.close()
