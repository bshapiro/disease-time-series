import numpy as np
import sys
from load_data import load_data
from pomegranate import NormalDistribution, HiddenMarkovModel
from khmm import df_to_sequence_list, cluster, init_gaussian_hmm


def init():
    m = 500  # restricts number of genes, used for local testing, None for all
    gc, mt, track = load_data(m)

    # khmm clustering over a range of k and states-per model
    k_range = [10, 25, 50, 100, 200]
    state_range = [5, 10, 25, 50, 100]

    msequences, mlabels = df_to_sequence_list(mt.data)
    gsequences, glabels = df_to_sequence_list(gc.data)

    sequences = np.concatenate((msequences, gsequences), 0)
    labels = np.concatenate((mlabels, glabels))

    sequences = np.concatenate((sequences, -1 * sequences))

    # tie positive and negative expression sequences
    tied = {}
    for i, label in enumerate(labels):
        tied[label] = [i, i+labels.size]

    labels = np.concatenate(((labels + '+'), (labels + '-')))

    # noise model trained on all data once
    # genes/metabolites will be assigned to noise model if other models
    # fail to model it better
    noise_dist = [NormalDistribution(0, 1)]
    noise_trans = np.array([[1]])
    starts = np.array([1])
    noise = HiddenMarkovModel.from_matrix(noise_trans, noise_dist, starts,
                                          name='noise')
    noise.freeze_distributions()

    return sequences, labels, tied, noise, k_range, state_range


def rand_init_invert_vit():
    sequences, labels, tied, noise, k_range, state_range = init()
    for n in state_range:
        for k in k_range:
            try:
                # directory to save files to
                odir_base = '../results/khmm/viterbi/rand_init_invert'
                collection_id = 'k-' + str(k) + '_n-' + str(n) + '_rand_init'
                odir = odir_base + '/' + collection_id

                print 'Learning: ', collection_id

                # generate random initial assignments
                # initialize models on random assignments
                randassign = np.random.randint(k, size=len(tied.keys()))
                assignments = {}
                models = {}
                for i in range(k):
                    model_id = str(i)
                    assignments[model_id] = \
                        np.where(randassign == i)[0].tolist()
                    in_model = assignments[model_id]
                    models[model_id] = \
                        init_gaussian_hmm(sequences[in_model, :], n, model_id)

                # add noise model
                models['noise'] = noise
                assignments['noise'] = []

                # all are un-fixed
                fixed = {}
                for model_id, model in models.iteritems():
                    fixed[model_id] = []

                # perform clustering
                models, assignments, c = cluster(models=models,
                                                 sequences=sequences,
                                                 assignments=assignments,
                                                 labels=labels,
                                                 algorithm='viterbi',
                                                 fixed=fixed, tied=tied,
                                                 odir=odir)
            except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()


def rand_init_invert():
    sequences, labels, tied, noise, k_range, state_range = init()
    for n in state_range:
        for k in k_range:
            try:
                # directory to save files to
                odir_base = '../results/khmm/rand_init_invert'
                collection_id = 'k-' + str(k) + '_n-' + str(n) + '_rand_init'
                odir = odir_base + '/' + collection_id

                print 'Learning: ', collection_id

                # generate random initial assignments
                # initialize models on random assignments
                randassign = np.random.randint(k, size=len(tied.keys()))
                assignments = {}
                models = {}
                for i in range(k):
                    model_id = str(i)
                    assignments[model_id] = \
                        np.where(randassign == i)[0].tolist()
                    in_model = assignments[model_id]
                    models[model_id] = \
                        init_gaussian_hmm(sequences[in_model, :], n, model_id)

                # add noise model
                models['noise'] = noise
                assignments['noise'] = []

                # all are un-fixed
                fixed = {}
                for model_id, model in models.iteritems():
                    fixed[model_id] = []

                # perform clustering
                models, assignments, c = cluster(models=models,
                                                 sequences=sequences,
                                                 assignments=assignments,
                                                 labels=labels,
                                                 algorithm='baum-welch',
                                                 fixed=fixed, tied=tied,
                                                 odir=odir)
            except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()
