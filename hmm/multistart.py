import numpy as np
import sys
from load_data import load_data
import time
from pomegranate import NormalDistribution, HiddenMarkovModel
from khmm import (df_to_sequence_list, cluster, init_gaussian_hmm, init_lr_hmm,
                  init_cycle_hmm, total_log_prob)


def init(m, seed):
    if m == -1:
        m = None
    gc, mt, track = load_data(m, seed)

    msequences, mlabels = df_to_sequence_list(mt.data)
    gsequences, glabels = df_to_sequence_list(gc.data)

    sequences = np.concatenate((msequences, gsequences), 0)
    labels = np.concatenate((mlabels, glabels))

    # noise model trained on all data once
    # genes/metabolites will be assigned to noise model if other models
    # fail to model it better
    noise_dist = [NormalDistribution(0, 1)]
    noise_trans = np.array([[1]])
    starts = np.array([1])
    noise = HiddenMarkovModel.from_matrix(noise_trans, noise_dist, starts,
                                          name='noise')
    noise.freeze_distributions()

    return sequences, labels, noise


def multistart(x, algorithm, initialization, m, k, n):

    if initialization == 'rand':
        init_method = init_gaussian_hmm
        init_args = {'n_states': n[0]}
    if init_lr_hmm == 'lr':
        init_method = init_lr_hmm
        s = n[0]
        sps = n[1]
        init_args = {'steps': s, 'state_per_step': sps, 'force_end': True}
    if initialization == 'cycle':
        init_method = init_cycle_hmm
        s = n[0]
        sps = n[1]
        init_args = {'steps': s, 'state_per_step': sps}

    sequences, labels, noise = init(m, 0)
    best = 0
    best_score = -1e1000
    for x in range(x):
        #try:
        # directory to save files to
        odir_base = '../results/khmm/multistart/' + str(m) + '/' + \
            initialization + '_k' + str(k) + 'n' + str(n) + '_' + algorithm
        collection_id = 'k' + str(k) + 'n' + str(n) + '_' + \
            initialization + '_' + str(x)
        odir = odir_base + '/' + str(x)

        print 'Learning: ', collection_id

        # generate random initial assignments
        # initialize models on random assignments
        np.random.seed(int(time.time()) + x)
        randassign = np.random.randint(k, size=labels.size)
        assignments = {}
        models = {}
        for i in range(k):
            model_id = str(i)
            assignments[model_id] = \
                np.where(randassign == i)[0].tolist()
            in_model = assignments[model_id]
            models[model_id] = \
                init_method(sequences[in_model, :], model_id=model_id, **init_args)

        # add noise model
        models['noise'] = noise
        assignments['noise'] = []

        # perform clustering
        models, assignments, c = cluster(models=models,
                                         sequences=sequences,
                                         assignments=assignments,
                                         algorithm=algorithm,
                                         labels=labels,
                                         odir=odir)

        score = total_log_prob(models, sequences, assignments)
        if best_score < score:
            best_score = score
            best = x

        """except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            print >> f, "Unexpected error:", sys.exc_info()[0]
            f.close()"""
    return best


if __name__ == "__main__":
    x = int(sys.argv[1])
    alg = sys.argv[2]
    initializtion = sys.argv[3]
    m = int(sys.argv[4])
    k = int(sys.argv[5])
    n = [int(i) for i in sys.argv[6:]]
    print alg
    print m
    print k
    print n
    # x, algorithm, initialization, m, k, n
    multistart(x, alg, initializtion, m, k, n)
