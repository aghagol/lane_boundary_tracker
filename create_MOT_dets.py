"""
This script creates an MOT-like dataset
Input format: a CSV with columns: longitude, latitude, altitude, timestamp
MOT format: a CSV with columns: frame#, -1, x, y, w, h, confidence, -1, -1, -1
"""
import os
from PIL import Image
import numpy as np

from MOT_util import make_MOT_det

if __name__ == "__main__":

  param = {}
  param['pose_step'] = 10
  param['image_nrows'] = 1000
  param['image_ncols'] = 1000
  param['object_size'] = 100
  param['pose_shifts'] = [(0,0),(-10,0),(10,0)]

  data_dir = '/home/mo/Desktop/HADComplianceGroundTruth1/'
  output_dir = './out/'

  tmp = np.zeros((param['image_nrows'],param['image_ncols']))

  print('Working on %s...'%data_dir)
  for drive in os.listdir(data_dir):
    if os.path.isdir(data_dir+drive):
      print('Working on drive %s'%drive)
      counter = 0
      for input_file in os.listdir(data_dir+drive):
        if input_file.endswith('pose.csv'):
          counter +=1
          if counter>1:
            print('Warning: found more than one pose csv in drive %s!'%drive)
            print('Processing the first one only...')
          else:
            det_out = output_dir+'%s/det/'%drive
            os.makedirs(det_out)
            make_MOT_det(data_dir+drive+'/'+input_file, det_out+'det.txt', param)

            img_out = output_dir+'%s/img1/'%drive
            os.makedirs(img_out)
            Image.fromarray(tmp).convert('RGB').save(img_out+'000001.jpg')



