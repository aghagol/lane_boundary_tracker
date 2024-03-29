#!/usr/bin/env python
"""
This script breaks a large drive-based JSON into small surface-based JSONs
"""
print(__doc__)

import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input",help="path to input drive JSONs")
parser.add_argument("--output",help="output path to save surface JSONs")
args = parser.parse_args()
data_dir = args.input+'/'
json_out_dir = args.output+'/'

json_list = [i for i in os.listdir(data_dir) if i.endswith('json')]
json_key = 'fromPosePointToSamplePoints'

for json_file in json_list:

  drive_id = json_file.split('-')[0]

  print('Working on drive %s'%(drive_id))
  os.makedirs(json_out_dir+drive_id)

  print('...loading data from %s'%(data_dir+json_file))
  with open(data_dir+json_file) as fjson:
    data = json.load(fjson)
  print('...done')
  
  surface_timestamp = {}
  surface_remove = set()
  for k,v in data[json_key].iteritems():
    for p in v['samplepoints']:
      if p['roadSurfaceID'] in surface_timestamp:
        surface_timestamp[p['roadSurfaceID']].add(k)
      else:
        surface_timestamp[p['roadSurfaceID']] = set([k])
      if p['boundaryCurveID'][:4]=='null' and p['roadSurfaceID'] not in surface_remove:
        print('WARNING: marking road surface %016d for removal because I found null lane boundaries in it!'%(int(p['roadSurfaceID'])))
        surface_remove.add(p['roadSurfaceID'])

  for surface, timestamp_set in surface_timestamp.iteritems():
    if surface in surface_remove:
      print('Skipping road surface %016d'%(int(surface)))
      continue
    start_timestamp = np.min([int(i) for i in timestamp_set])
    with open(json_out_dir+drive_id+'/surface_%d_%016d.json'%(start_timestamp,int(surface)),'w') as fjson:
      json.dump({json_key:{k:v for k,v in data[json_key].iteritems() if k in timestamp_set}},fjson,indent=4)
