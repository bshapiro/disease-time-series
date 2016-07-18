import numpy as np
import sys
from load_data import load_data
from pomegranate import NormalDistribution, HiddenMarkovModel
from khmm import df_to_sequence_list, cluster, init_gaussian_hmm

m = 500  # restricts number of genes, used for local testing, None for all
gc, mt, track = load_data(m)

# khmm clustering over a range of k and states-per model
k_range = [10, 25, 50, 100, 200]
state_range = [5, 10, 25, 50, 100]

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

max_iter = 500  # max iterations
eps = 1e-6  # convergence threshold
k = k_range[0]
n = state_range[0]
d_range = [0, 1, 5, 10, 15, 30]

# oassignments = np.random.randint(k+1, size=labels.size)
# fixed = np.array([0] * oassignments.size)
for n in state_range:
    for k in k_range:
        try:
            odir_base = 'rand_init'  # directory to save files to
            collection_id = 'k-' + str(k) + '_n-' + str(n) + '_rand_init'
            odir = odir_base + '/' + collection_id

            print 'Learning: ', collection_id

            # generate random initial assignments
            # initialize models on random assignments
            randassign = np.random.randint(k, size=labels.size)
            assignments = {}
            models = {}
            for i in range(k):
                model_id = str(i)
                assignments[model_id] = np.where(randassign == i)[0].tolist()
                in_model = assignments[model_id]
                models[model_id] = init_gaussian_hmm(sequences[in_model, :],
                                                     n, model_id)

            # add noise model
            models['noise'] = noise
            assignments['noise'] = []

            # all are un-fixed
            fixed = np.array([0] * labels.size)

            # perform clustering
            models, assignments, c = cluster(models=models,
                                             sequences=sequences,
                                             assignments=assignments,
                                             labels=labels, fixed=fixed,
                                             eps=eps, max_it=max_iter,
                                             odir=odir)

        except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            print >> f, "Unexpected error:", sys.exc_info()[0]
            f.close()
