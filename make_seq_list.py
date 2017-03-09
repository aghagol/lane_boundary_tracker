""" 
Create a list of sequences as input to disp.py

CSV input format is: 3 columns 
"name" (seq_name), "dpath" (detection file path), "tpath" (track file path)
"""

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
seqs_df.to_csv(output_filename,index=False)