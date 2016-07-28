from templates import lr_hmm
from load_data import load_data
from pickle import load
from templates import lr_hmm
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
ALGORITHM = 'viterbi'

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
        print 'No gene file specified, using default.'
        odir = OD

    gc, mt, track = load_data()
    genes = load(open(genefile,  'r'))

    data = pd.concat((gc.data.loc[genes, :], mt.data))
    sequences = data.as_matrix()

    model = lr_hmm(sequences, data.columns.size, states_per_step,
                   self_trans=False, force_end=True, model_id='Profile HMM',
                   seed=None)

    model.fit(sequences.astype(float),
              verbose=SHOW_TRAINING,
              algorithm=ALGORITHM,
              n_jobs=THREADCOUNT)

    assignments = defaultdict(list)
    paths = {}
    for sequence in data.index.values:
        path = tuple(model.predict(data.loc[sequence, :]))
        print path
        assignments[path].append(sequence)

    # write cluster assignments
    if not os.path.isdir(odir):
        os.makedirs(odir)
    filepath = odir.split('/') + ['assignments.txt']
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
    filepath = odir.split('/') + ['model']
    filepath = '/'.join(filepath)
    with open(filepath, 'w') as f:
        f.write(model.to_json())

print len(assignments)
