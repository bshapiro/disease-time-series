from templates import lr_hmm
from load_data import load_data
from pickle import load
import time
from collections import defaultdict
import pandas as pd
import numpy as np
import sys
import os

SPS = 3
GF = None
OD = './'
THREADCOUNT = 2
SHOW_TRAINING = True
RESTARTS = 4
ALGORITHM = 'baum-welch'

if __name__ == '__main__':
    try:
        states_per_step = int(sys.argv[1])
    except:
        print 'Using default states per step = ', SPS
        states_per_step = SPS
    try:
        genefile = sys.argv[2]
    except:
        print 'No gene file specified, using default.'
        genefile = GF
    try:
        odir = sys.argv[3]
    except:
        print 'No output path specified, using default.'
        odir = OD
    try:
        restarts = int(sys.argv[4])
    except:
        print 'No restart number specified, using default: ', RESTARTS
        restarts = RESTARTS

    gc, mt, track = load_data()
    genes = load(open(genefile,  'r'))

    data = (gc.data.loc[genes, :])
    sequences = data.as_matrix()

    for x in range(restarts):
        out_directory = odir.split('/') + [str(x)]
        out_directory = '/'.join(out_directory)
        model = lr_hmm(sequences, data.columns.size, states_per_step,
                       self_trans=False, force_end=True,
                       model_id='Profile HMM',
                       seed=int(time.time()))

        model.fit(sequences.astype(float),
                  verbose=SHOW_TRAINING,
                  algorithm=ALGORITHM,
                  n_jobs=THREADCOUNT)

        # standard assignment scheme
        assignments = defaultdict(list)
        paths = {}
        for sequence in data.index.values:
            path = tuple(model.predict(data.loc[sequence, :]))
            print path
            assignments[path].append(sequence)

        # agglomerative clustering scheme

        # write cluster assignments
        if not os.path.isdir(out_directory):
            os.makedirs(out_directory)
        filepath = out_directory.split('/') + ['assignments.txt']
        filepath = '/'.join(filepath)
        with open(filepath, 'w') as f:
            for path, members in assignments.iteritems():
                f.write(str(path))
                f.write('\n')
                f.write('\t'.join(members))
                f.write('\n')
                f.write('\n')
                f.write('\n')

        # write model json
        filepath = out_directory.split('/') + ['model']
        filepath = '/'.join(filepath)
        with open(filepath, 'w') as f:
            f.write(model.to_json())

print len(assignments)
