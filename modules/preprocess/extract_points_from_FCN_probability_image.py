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
parser.add_argument("--input",      help="path to CNN predictions")
parser.add_argument("--output",     help="path to output fuse files")
parser.add_argument("--poses",      help="path to pose CSV files")
parser.add_argument("--config",     help="path to config file")
parser.add_argument("--drives",     help="path to drives list text-file")
parser.add_argument("--verbosity",  help="verbosity level", type=int)
args = parser.parse_args()

if args.verbosity>=2:
  print(__doc__)

if not os.path.exists(args.output):
  os.makedirs(args.output)

#read preprocessing parameters from the configuration JSON file
with open(args.config) as fparam:
  parameters = json.loads(jsmin(fparam.read()))["preprocess"]

#get names of drives to be processed
drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

#format of output .fuse files. 
fuse_fmt = ['%.16f','%.16f'] #format: latitude, longitude

for drive in drive_list:
  if args.verbosity>=2:
    print('Working on drive %s'%drive)

  #read metadata (in CSV format) for images of the current drive
  #meta column labels: name, time_start, time_end, min_lat, min_lon, max_lat, max_lon
  meta = pd.read_csv(os.path.join(args.input,drive,'meta.csv'), skipinitialspace=True)

  for n,res in meta.iterrows():

    #generate an image tag (must uniquely identify each image) to include in the .fuse file name
    image_tag = '%d'%(res['time_start'])

    #skip the .fuse file generation if the .fuse file exists already
    if os.path.exists(args.output+'/'+drive+'_'+image_tag+'.png.fuse'): continue

    #skip the .fuse file generation if the image does not exist
    if not os.path.exists(os.path.join(args.input,drive,res['name'])): continue

    if args.verbosity>=2:
      print('\tworking on %s'%(res['name']))

    #read the probability image (FCN output)
    pred_im = misc.imread(os.path.join(args.input,drive,res['name']))/(2.**16-1)

    if args.verbosity>=2:
      print('\t\timage size: %d x %d'%(pred_im.shape[0],pred_im.shape[1]))

    #find peaks using Ian's "peak_finding" algorithm and discard non-peak detections
    if parameters['ians_peak_filter']:
      peaks = peak_finding.peaks_clean(pred_im, 0.3, input_as_mag=True)
      pred_im = pred_im*peaks

    #discard detections that are not local maxima within a "maxpool_size" window
    pred_im[pred_im!=maximum_filter(pred_im,size=parameters['maxpool_size'])]=0

    #only retain high confidence detection points
    ok = pred_im>parameters['confidence_thresh']
    
    #get lat-lon coordiantes of detection points
    bbox = np.r_[res['min_lat'], res['min_lon'], res['max_lat'], res['max_lon']]
    loc_pix = np.c_[np.nonzero(ok)] #row and column pixel indices
    loc_im = loc_pix.astype(float)/pred_im.shape #scale to [0,1] in each direction
    lat_lon = (loc_im[:,::-1])*(bbox[2:]-bbox[:2]) + bbox[:2] #final lat-lon coordinates

    if args.verbosity>=2:
      print('\t\trecorded %d detection points from %s'%(lat_lon.shape[0],res['name']))

    #save detection points in .fuse files
    np.savetxt(args.output+'/'+drive+'_'+image_tag+'.png.fuse',lat_lon,fmt=fuse_fmt)
