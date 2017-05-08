import os
from shutil import copyfile

pose_path = '/media/mo/remoteTitan/dataset_on_titan-pascal/topdown_urban/4-27-rotated/meta/'
destination_dir = '/home/mo/data_lifeng/sampled_training_data_5-4/poses/'
if not os.path.exists(destination_dir):
  os.mkdir(destination_dir)

drive_ids = [i for i in os.listdir(pose_path) if i[:2]=='HT']
for drive in drive_ids:
  pose_file = drive+'-pose.csv'
  copyfile(pose_path+drive+'/'+pose_file,destination_dir+pose_file)

