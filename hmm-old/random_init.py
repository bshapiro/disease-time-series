import addpath
import numpy as np
from pickle import dump
from hmmlearn import hmm
from src.khmm import df_to_sequence_list, cluster
from load_data import gc, mt

# m = gc.data.index.size  # restricts number of genes, used for local testing
m = 500
# khmm clustering over a range of k and states-per model
k_range = [10, 50, 100, 150, 200, 250, 300, 500]
state_range = [5, 10, 20, 30, 40, 50, 60]

msequences, mlengths, mlabels = df_to_sequence_list(mt.data)
gsequences, glengths, glabels = df_to_sequence_list(gc.data.iloc[:m, :])

sequences = np.concatenate((msequences, gsequences))
lengths = np.concatenate((mlengths, glengths))
labels = np.concatenate((mlabels, glabels))

# noise model trained on all data once
# genes/metabolites will be assigned to noise model if other models
# fail to model it better
noise = hmm.GaussianHMM(n_components=1)
noise.fit(sequences, lengths)

max_iter = 200  # max iterations
eps = 1e-5  # convergence threshold
k = k_range[0]
n = state_range[0]

odr = 'rand_init'  # directory to save files to
model_id = 'k-' + str(k) + '_n-' + str(n) + '_rand_init'
print 'Learning: ', model_id
models = np.empty(0)
for i in range(k):
    models = np.append(models, hmm.GaussianHMM(n_components=n))

assignments = np.random.randint(k+1, size=lengths.size)
fixed = np.array([0] * assignments.size)

save_name = model_id + '.' + 'cluster.iter'
models, assignments, converged = cluster(models, np.array([noise]),
                                         sequences, lengths,
                                         assignments, fixed, eps,
                                         max_iter,
                                         save_name=model_id,
                                         odir=odr)
# save text output
save_name = model_id + '.' + 'cluster.results'
save_name = odr.split('/') + save_name.split('/')
save_name = '/'.join(save_name)
f = open(save_name, 'w')
print >> f, 'ASSIGNMENTS: '
print >> f, str(assignments)
for i, model in enumerate(models):
    print >> f, 'MODEL ', i, ': '
    print >> f, 'start_prob = \n', str(model.startprob_)
    print >> f, 'transition_matrix = \n', str(model.transmat_)
    print >> f, 'means = \n', str(model.means_)
    print >> f, 'covars = \n', str(model.covars_)

# save pickle
save_name = model_id + '.p'
save_name = odr.split('/') + save_name.split('/')
save_name = '/'.join(save_name)
dump([labels, assignments, lengths, fixed, models, noise, eps,
      converged], open(save_name, 'wb'))

for k in k_range:
    for n in state_range:
        odr = 'rand_init'  # directory to save files to
        model_id = 'k-' + str(k) + '_n-' + str(n) + '_rand_init'
        print 'Learning: ', model_id
        models = np.empty(0)
        for i in range(k):
            models = np.append(models, hmm.GaussianHMM(n_components=n))

        assignments = np.random.randint(k+1, size=lengths.size)
        fixed = np.array([0] * assignments.size)
        try:
            save_name = model_id + '.' + 'cluster.iter'
            models, assignments, converged = cluster(models, np.array([noise]),
                                                     sequences, lengths,
                                                     assignments, fixed, eps,
                                                     max_iter,
                                                     save_name=model_id,
                                                     odir=odr)
            # save text output
            save_name = model_id + '.' + 'cluster.results'
            save_name = odr.split('/') + save_name.split('/')
            save_name = '/'.join(save_name)
            f = open(save_name, 'w')
            for i, model in enumerate(models):
                print >> f, 'ASSIGNMENTS: '
                print >> f, str(assignments)
                print >> f, 'MODEL ', i, ': '
                print >> f, 'start_prob = ', str(model.startprob_)
                print >> f, 'transition_matrix = ', str(model.transmat_)
                print >> f, 'means = ', str(model.means_)
                print >> f, 'covars = ', str(model.covars_)

            # save pickle
            save_name = model_id + '.p'
            save_name = odr.split('/') + save_name.split('/')
            save_name = '/'.join(save_name)
            dump([labels, assignments, lengths, fixed, models, noise, eps,
                  converged], open(save_name, 'wb'))
        except:
            error_file = odr.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', model_id
            f.close()
