import numpy as np
from load_data import load_data
from pomegranate import NormalDistribution, HiddenMarkovModel
from khmm import df_to_sequence_list, cluster, init_gaussian_hmm

m = 500  # restricts number of genes, used for local testing, None for all
gc, mt, track = load_data(m)

# khmm clustering over a range of k and states-per model
k_range = [10, 50, 100, 200, 500]
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
noise = HiddenMarkovModel.from_matrix(noise_trans, noise_dist, starts)


max_iter = 500  # max iterations
eps = 1e-6  # convergence threshold
k = k_range[0]
n = state_range[0]

for k in k_range:
    for n in state_range:
        odir_base = 'rand_init'  # directory to save files to
        collection_id = 'k-' + str(k) + '_n-' + str(n) + '_rand_init'
        odir = odir_base + '/' + collection_id

        print 'Learning: ', collection_id

        assignments = np.random.randint(k+1, size=labels.size)
        fixed = np.array([0] * assignments.size)

        try:
            # initialize models
            models = np.empty(0)
            for i in range(k):
                in_model = np.where(assignments == i)[0]
                model = init_gaussian_hmm(sequences[np.where(assignments == 0)
                                          [0], :],
                                          n, model_id=str(i))
                models = np.append(models, model)

            models, assignments, c = cluster(models=models,
                                             noise_models=np.array([noise]),
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
            f.close()
