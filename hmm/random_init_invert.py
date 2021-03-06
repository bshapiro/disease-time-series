import numpy as np
import sys
from load_data import load_data
from pomegranate import NormalDistribution, HiddenMarkovModel
from khmm import df_to_sequence_list, cluster, init_gaussian_hmm


def init(m, seed):
    if m == -1:
        m = None
    gc, mt, track = load_data(m, seed)

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

    return sequences, labels, tied, noise


def rand_init_invert(algorithm, m, k, state_range):
    sequences, labels, tied, noise = init(m, 0)
    for n in state_range:
        try:
            # directory to save files to
            odir_base = '../results/khmm/' + algorithm + '/' + str(m) + \
                        '/rand_init_invert'
            collection_id = 'm' + str(m) + 'k' + str(k) + 'n' + str(n) + \
                            '_rand_init_invert'
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
                                             algorithm=algorithm,
                                             fixed=fixed, tied=tied,
                                             odir=odir)
        except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            print >> f, "Unexpected error:", sys.exc_info()[0]
            f.close()


if __name__ == "__main__":
    alg = sys.argv[1]
    m = int(sys.argv[2])
    k = int(sys.argv[3])
    state_range = [int(i) for i in sys.argv[4:]]
    print alg
    print m
    print k
    print state_range
    rand_init_invert(alg, m, k, state_range)
