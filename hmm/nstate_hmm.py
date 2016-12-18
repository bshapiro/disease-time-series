from templates import gaussian_hmm
from load_data import load_data
from pickle import load, dump
import time
from collections import defaultdict
import pandas as pd
import numpy as np
import sys
import os

STATES = 20
LOW = -2
HIGH = 2
VAR = 0.5
GF = None
OD = './'
THREADCOUNT = 4
SHOW_TRAINING = False
RESTARTS = 4
ALGORITHM = 'baum-welch'
# ALGORITHM = 'viterbi'
STOP_THRESHOLD = 1e-8

if __name__ == '__main__':
    try:
        n_states = int(sys.argv[1])
        print 'states: ', n_states
    except:
        print 'Using default number of states = ', STATES
        n_states = STATES
    try:
        lower = int(sys.argv[2])
    except:
        print 'Using default lower = ', LOW
        lower = STATES
    try:
        upper = int(sys.argv[3])
    except:
        print 'Using default upper = ', HIGH
        upper = HIGH
    try:
        var = int(sys.argv[4])
    except:
        print 'Using default var = ', VAR
        var = VAR
    try:
        genefile = sys.argv[5]
        print 'genes: ', genefile
    except:
        print 'No gene file specified, using default.'
        genefile = GF
    try:
        odir = sys.argv[6]
    except:
        print 'No output path specified, using default.'
        odir = OD
    try:
        restarts = int(sys.argv[7])
        print 'restarts: ', restarts
    except:
        print 'No start index number specified, using default: ', RESTARTS
        index = RESTARTS
    try:
        data_file = sys.argv[8]
        print 'data: ', data_file
    except:
        pass

    data = pd.DataFrame.from_csv(data_file, sep=' ')
    genes = load(open(genefile,  'r'))
    data = (data.loc[genes, :])
    data = ((data.T - data.T.mean()) / data.T.std()).T
    sequences = data.as_matrix()

    for x in range(restarts):
        if restarts == 1:
            out_directory = odir
        else:
            out_directory = odir.split('/') + [str(x)]
            out_directory = '/'.join(out_directory)

        model = gaussian_hmm(n_states=n_states, lower=lower, upper=upper,
                             variance=var, model_id=('n-state HMM:' + str(x)))

        model.fit(sequences.astype(float),
                  verbose=SHOW_TRAINING,
                  stop_threshold=STOP_THRESHOLD,
                  algorithm=ALGORITHM,
                  n_jobs=THREADCOUNT)

        # standard assignment scheme
        assignments = defaultdict(list)
        paths = {}
        for sequence in data.index.values:
            path = tuple(model.predict(data.loc[sequence, :]))
            print path
            assignments[path].append(sequence)

        if not os.path.isdir(out_directory):
            os.makedirs(out_directory)

        # save cluster dictionary to pickle
        filepath = out_directory.split('/') + ['assignments.p']
        filepath = '/'.join(filepath)
        dump(assignments, open(filepath, 'w'))

        # write cluster assignments
        filepath = out_directory.split('/') + ['assignments.txt']
        filepath = '/'.join(filepath)
        with open(filepath, 'w') as f:
            for path, members in assignments.iteritems():
                f.write(str(path))
                f.write('\n')
                f.write('\t'.join(members))
                f.write('\n')

        # write model json
        filepath = out_directory.split('/') + ['model']
        filepath = '/'.join(filepath)
        with open(filepath, 'w') as f:
            f.write(model.to_json())

print len(assignments)
