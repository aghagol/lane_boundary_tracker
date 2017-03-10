"""
This script creates standard MOT detections file
input: a CSV with columns: longitude, latitude, altitude, timestamp
output: a CSV with columns: frame#, -1, x, y, w, h, confidence, -1, -1, -1
"""
import numpy as np
import pandas as pd

# posefile = '/home/mo/Desktop/HADComplianceGroundTruth1/HT010_1465209606/80416021-pose.csv'
posefile = '80416021-pose.csv'

pose_df = pd.read_csv(posefile,header=None,sep=' ')

# subsample the time and create frames

# maybe shift and repeat the trajectory




