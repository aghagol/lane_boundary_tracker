#!/usr/bin/env python
"""
This script generates points (fuse format) from CNN predictions
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

with open(args.config) as fparam:
  parameters = json.loads(jsmin(fparam.read()))["preprocess"]

drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

fuse_fmt = ['%.16f','%.16f']
for drive in drive_list:
  if args.verbosity>=2:
    print('Working on drive %s'%drive)

  meta = pd.read_csv(os.path.join(args.input,drive,'meta.csv'), skipinitialspace=True)

  for n,res in meta.iterrows():
    image_tag = '%d'%(res['time_start'])

    if os.path.exists(args.output+'/'+drive+'_'+image_tag+'.png.fuse'): continue
    if not os.path.exists(os.path.join(args.input,drive,res['name'])): continue
    if args.verbosity>=2:
      print('\tworking on %s'%(res['name']))

    pred_im = misc.imread(os.path.join(args.input,drive,res['name']))/(2.**16-1)
    if args.verbosity>=2:
      print('\t\timage size: %d x %d'%(pred_im.shape[0],pred_im.shape[1]))

    #find peaks and filter
    if parameters['peak_filter']:
      peaks = peak_finding.peaks_clean(pred_im, 0.3, input_as_mag=True)
      pred_im = pred_im*peaks

    pred_im[pred_im!=maximum_filter(pred_im,size=parameters['maxpool_size'])]=0

    #only retain high confidence detection points
    ok = pred_im>parameters['confidence_thresh']
    
    #get lat-lon coordiantes of detection points
    bbox = np.r_[res['min_lat'], res['min_lon'], res['max_lat'], res['max_lon']]
    loc_pix = np.c_[np.nonzero(ok)] # row and column pixel indices
    loc_im = loc_pix.astype(float)/pred_im.shape # scale to [0,1] in each direction
    lat_lon = (loc_im[:,::-1])*(bbox[2:]-bbox[:2]) + bbox[:2] # Final lat-lon coordinates
    if args.verbosity>=2:
      print('\t\trecording %d points'%(lat_lon.shape[0]))

    np.savetxt(args.output+'/'+drive+'_'+image_tag+'.png.fuse',lat_lon,fmt=fuse_fmt)
