""" 
Creating a list of sequences as input to disp.py

output CSV has 3 columns:
  name..............this is the sequence name
  dpath.............this is the detection file path
  tpath.............this is the track file path

"""
print(__doc__)

import pandas as pd

output_filename = 'seqs.csv'

phase = 'train'
sequences = [
  'PETS09-S2L1',
  'TUD-Campus',
  'TUD-Stadtmitte',
  'ETH-Bahnhof',
  'ETH-Sunnyday',
  'ETH-Pedcross2',
  'KITTI-13',
  'KITTI-17',
  'ADL-Rundle-6',
  'ADL-Rundle-8',
  'Venice-2',
]

seqs = {}
seqs['name'] = sequences
seqs['dpath'] = ['mot_benchmark/%s/%s/det/det.txt'%(phase,seq) for seq in sequences]
seqs['tpath'] = ['output/%s.txt'%(seq) for seq in sequences]

seqs_df = pd.DataFrame(seqs)
seqs_df = seqs_df[['name','dpath','tpath']] # sort

seqs_df.to_csv(output_filename,index=False,header=True)