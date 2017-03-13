"""
arrange chunks into continuous lane boundaries

"""

import numpy as np
import pandas as pd

# read data
drive_path = '/home/mo/Desktop/HADComplianceGroundTruth1/HT010_1465209606/'
chunks_path =  drive_path+'EarthcoreRoadFeatures/'

pose_filename = drive_path+'80416021-pose.csv'
pose_df = pd.read_csv(posefile,header=None,sep=' ')


