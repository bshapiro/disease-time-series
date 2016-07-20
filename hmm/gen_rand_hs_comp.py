import numpy as np
import sys
import time
from load_data import load_data
from pomegranate import NormalDistribution, HiddenMarkovModel
from khmm import df_to_sequence_list, cluster, init_gaussian_hmm


def init(m=None, seed=None):
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

    # khmm clustering over a range of k and states-per model
    k_range = [10, 25, 50, 100, 200]
    state_range = [5, 10, 25, 50, 100]
    return sequences, labels, noise, k_range, state_range


def gen(m, x):
    """
    generate x sets of models on m genes
    gene_seed is the random seed for gene selection
    assign_seed is the seed for random assignment generation
    model_seed is the random seed for model initialization
    x is a batch identifier, any output with x was generated from identicle
    random initializations
    """

    # random subset of sequences shared for baum-welch and viterbi training
    sequences, labels, noise, k_range, state_range = init(m, int(time.time()))
    k = 20
    n = 10

    # generate random initial assignments
    np.random.seed(int(time.time()) + 11)
    randassign = np.random.randint(k, size=labels.size)
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

    # copy assignments and models for different training regimes
    bm_models = {}
    vt_models = {}
    for model_id, model in models.iteritems():
        bm_models[model_id] = model.copy()
        vt_models[model_id] = model.copy()

    bm_assignments = assignments.copy()
    vt_assignments = assignments.copy()

    # directory to save files to
    odir_base = '../results/khmm/hard_soft_comp/'

    # train with viterbi algorithm (hard assignment)
    collection_id = 'vt-' + str(x)
    odir = odir_base + '/' + collection_id
    print 'Learning: ', collection_id
    # perform clustering
    models, assignments, c = cluster(models=vt_models,
                                     sequences=sequences,
                                     assignments=vt_assignments,
                                     algorithm='viterbi',
                                     labels=labels,
                                     odir=odir)

    # train with baum-welch (soft assignments)
    collection_id = 'bm-' + str(x)
    odir = odir_base + '/' + collection_id
    print 'Learning: ', collection_id
    # perform clustering
    models, assignments, c = cluster(models=bm_models,
                                     sequences=sequences,
                                     assignments=bm_assignments,
                                     algorithm='baum-welch',
                                     labels=labels,
                                     odir=odir)


if __name__ == "__main__":
    m = int(sys.argv[1])
    x = int(sys.argv[2])
    for i in range(x):
        gen(m, x)
