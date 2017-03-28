"""
This script creates an MOT-like dataset
Input format: a JSON file
MOT format: a CSV with columns: frame#, -1, x, y, w, h, confidence, -1, -1, -1
"""
import os
from PIL import Image
import numpy as np

from MOT_util import JSON_to_MOT_det

param = {
  'pixel_size':1., #in meters
  'object_size':10., #in meters
}

data_dir = 'jsons/'
output_dir = './out/'

skipped = 0
processed = 0
for drive in os.listdir(data_dir):
  if drive.endswith('json'):
    print('Working on drive %s'%drive)
    det_out = output_dir+'%s/det/'%drive.split('-')[0]
    os.makedirs(det_out)
    if JSON_to_MOT_det(data_dir+drive, det_out+'det.txt', param):
      print('\tDrive too large! Skipping...')
      skipped +=1
      os.rmdir(det_out)
      os.rmdir(output_dir+'%s/'%drive.split('-')[0])
    else: #create also the image
      img_out = output_dir+'%s/img1/'%drive.split('-')[0]
      os.makedirs(img_out)
      tmp = np.zeros((param['image_nrows'],param['image_ncols']))
      Image.fromarray(tmp).convert('RGB').save(img_out+'000001.jpg')
      processed +=1

print('Skipped=%d, Processed=%d'%(skipped,processed))


