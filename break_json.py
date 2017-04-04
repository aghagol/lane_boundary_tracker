"""
This script breaks a large drive-based JSON into small surface-based JSONs
Place drive-based JSONs in `drive_jsons` directory
Code outputs surface-based JSONs into `ss_jsons` directory
(code automatically creates subdirectories for each drive)
"""
print(__doc__)

import os
import json

data_dir = 'drive_jsons/'
json_out_dir = 'ss_jsons/'

json_list = [i for i in os.listdir(data_dir) if i.endswith('json')]

for json_file in json_list:

  drive_id = json_file.split('-')[0]

  print('Working on drive %s'%(drive_id))
  os.makedirs(json_out_dir+drive_id)

  print('...loading data from %s'%(data_dir+json_file))
  with open(data_dir+json_file) as fjson:
    data = json.load(fjson)
  print('...done')
  
  surface_timestamp = {}
  for k,v in data['fromPosePointToSamplePoints'].iteritems():
    for p in v['samplepoints']:
      if p['roadSurfaceID'] in surface_timestamp:
        surface_timestamp[p['roadSurfaceID']].append(k)
      else:
        surface_timestamp[p['roadSurfaceID']] = [k]

  for surface in surface_timestamp:
    with open(json_out_dir+drive_id+'/surface_%s.json'%(surface),'w') as fjson:
      json.dump({'fromPosePointToSamplePoints':{k:v for k,v in data['fromPosePointToSamplePoints'].iteritems() if k in surface_timestamp[surface]}},fjson,indent=4)