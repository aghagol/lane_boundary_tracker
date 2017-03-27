""" 
Creating a list of sequences as input to disp.py

output CSV has 3 columns:
  name..............this is the sequence name
  dpath.............this is the detection file path
  tpath.............this is the track file path

Make a symbolic link `data` to MOT dataset
Make a symbolic link `tracks` to output of tracker

"""
print(__doc__)

import pandas as pd
import os

output_filename = 'seqs.csv'

sequences = [i for i in os.listdir('data') if os.path.isdir('data/'+i)]
seq_set_trks = set([i[:-4] for i in os.listdir('tracks') if i.endswith('txt')])
seq_set_dets = set(sequences)
for seq in sequences:
    if not seq in seq_set_trks:
        seq_set_dets.remove(seq)
sequences = sorted(list(seq_set_dets))

seqs = {}
seqs['name'] = sequences
seqs['dpath'] = ['data/%s/det/det.txt'%(seq) for seq in sequences]
seqs['tpath'] = ['tracks/%s.txt'%(seq) for seq in sequences]

seqs_df = pd.DataFrame(seqs)
seqs_df = seqs_df[['name','dpath','tpath']] # sort

seqs_df.to_csv(output_filename,index=False,header=True)

