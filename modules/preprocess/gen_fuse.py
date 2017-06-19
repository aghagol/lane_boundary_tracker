#!/usr/bin/env python
"""
This script generates fuse files from CNN predictions (images)
"""
print(__doc__)

import pandas as pd
import numpy as np
from scipy import misc
import os
import argparse
import json
from jsmin import jsmin

import peak_finding

parser = argparse.ArgumentParser()
parser.add_argument("--input",  help="path to data root directory")
parser.add_argument("--images", help="path to CNN predictions")
parser.add_argument("--output", help="path to output fuse files")
parser.add_argument("--poses",  help="path to pose CSV files")
parser.add_argument("--config", help="path to config file")
parser.add_argument("--drives", help="path to drives list text-file")
args = parser.parse_args()
input_path = args.input+'/'
images_path = args.input+'/'
output_path = args.output+'/'
poses_path = args.poses+'/'

if not os.path.exists(output_path):
  os.makedirs(output_path)

with open(args.config) as fparam:
  parameters = json.loads(jsmin(fparam.read()))["preprocess"]

drive_list = []
with open(args.drives) as fdrivelist:
  for line in fdrivelist:
    drive_list.append(line.strip())

fuse_fmt = ['%.16f','%.16f']
for drive in drive_list:
  print('Working on drive %s'%drive)

  meta = pd.read_csv(os.path.join(input_path,'meta.csv'), skipinitialspace=True)

  for n,res in meta.iterrows():
    image_tag = '%d'%((res['time_start']+res['time_end'])/2)
    if os.path.exists(output_path+drive+'_'+image_tag+'.png.fuse'): continue
    if not os.path.exists(os.path.join(images_path,res['name'])): continue
    print('\tworking on %s'%(res['name']))

    pred_im = misc.imread(os.path.join(images_path,res['name']))/(2.**16-1)
    peaks = peak_finding.peaks_clean(pred_im, 0.3, input_as_mag=True)
    pred_im = pred_im*peaks
    ok = pred_im>0.99 # Choose your own threshold -- this isn't necessarily a good one.
    bbox = np.r_[res['min_lat'], res['min_lon'], res['max_lat'], res['max_lon']]
    loc_pix = np.c_[np.nonzero(ok)] # row and column pixel indices
    loc_im = loc_pix.astype(float)/pred_im.shape # scale to [0,1] in each direction
    lat_lon = (loc_im[:,::-1])*(bbox[2:]-bbox[:2]) + bbox[:2] # Final lat-lon coordinates
    print('\t\t%d points'%(lat_lon.shape[0]))
    
    np.savetxt(output_path+drive+'_'+image_tag+'.png.fuse',lat_lon,fmt=fuse_fmt)
