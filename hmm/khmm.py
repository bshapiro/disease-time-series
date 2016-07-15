from pomegranate import NormalDistribution, HiddenMarkovModel
import math
import time
import numpy as np
import os

THREADCOUNT = 12
ALGORITHM = 'baum-welch'
SHOW_TRAINING = True


def cluster(models, noise_models, sequences, assignments, labels, fixed, eps,
            max_it, odir='./'):

    # open path to output file
    filepath = odir.split('/') + ['iteration_report.txt']
    filepath = '/'.join(filepath)
    try:
        f = open(filepath, 'w')
    except:
        directory = '/'.join(filepath.split('/')[:-1])
        print "Creating directory...", directory
        os.makedirs(directory)
        f = open(filepath, 'w')
    f.close()

    # intial probability calculation
    curr_log_prob = total_log_prob(models, noise_models, sequences,
                                   assignments)
    start_time = time.time()

    with open(filepath, 'a') as f:
        print >> f, 'Init', ', Log Prob = ', str(curr_log_prob), \
              ', Assignments = ', str(np.bincount(assignments)), \
              ', time-elapsed = 00:00:00'
    # iterative model assignment
    iteration = 0
    prior_assignments = np.empty([])
    delta = eps
    while iteration <= max_it:

        prior_models = np.array([models[i].copy() for i in range(models.size)])
        prior_assignments = np.copy(assignments)

        assignments = assign(models, noise_models, sequences,
                             assignments, fixed)

        train(models, sequences, assignments)

        new_log_prob = total_log_prob(models, noise_models,
                                      sequences, assignments)
        delta = new_log_prob - curr_log_prob
        curr_log_prob = new_log_prob
        time_elapsed = time.strftime("%H:%M:%S",
                                     time.gmtime(time.time() - start_time))

        with open(filepath, 'a') as f:
            print >> f, 'Iter = ', str(iteration), \
                  'Delta = ', str(delta), ', Log Prob: ', str(curr_log_prob), \
                  ', Assignments= ', str(np.bincount(assignments)), \
                  'time elapsed = ', time_elapsed

        if delta < 0 and eps > 0:
            models = prior_models
            assignments = prior_assignments
            with open(filepath, 'a') as f:
                print >> f, 'Local optimum found at iter: ', (iteration - 1)
            break
        elif delta < eps or np.all(prior_assignments != assignments):
            with open(filepath, 'a') as f:
                print >> f, 'Local optimum found at iter: ', (iteration)
            break

        iteration += 1

    # write conlusive lines in report
    converged = (delta < eps)
    if not converged:
        if (iteration > max_it):
            line = 'Iteration limit reached: ' + str(max_it)
        else:
            line = 'Something went wrong'

        with open(filepath, 'a') as f:
            f.write(str(line))

    # write json representations of models
    for model in models:
        filepath = odir.split('/') + [model.name]
        filepath = '/'.join(filepath)
        with open(filepath, 'w') as f:
            f.write(model.to_json())

    # write cluster assignments
    filepath = odir.split('/') + ['assignments.txt']
    filepath = '/'.join(filepath)
    with open(filepath, 'w') as f:
        for i, model in enumerate(models):
            f.write(model.name)
            f.write('\n')
            f.write('\t'.join(labels[np.where(assignments == i)]))
            # f.write(str(labels[np.where(assignments == i)]))
            f.write('\n')

    return models, assignments, converged


def train(models, sequences, assignments):

    # train the models based on current assignment
    for i, model in enumerate(models):
        # prior_model = models[i].copy()
        in_model = np.where(assignments == i)[0]
        if in_model.size != 0:
            sequence_set = sequences[in_model, :]
            model.thaw_distributions()
            model.fit(sequence_set, verbose=SHOW_TRAINING, algorithm=ALGORITHM,
                      n_jobs=THREADCOUNT)
            # if isnan(model.fit(sequence_set)):
            #    models[i] = prior_model


def assign(models, noise_models, sequences, assignments, fixed):
    # import pdb; pdb.set_trace()
    scores = score_matrix(models, noise_models, sequences,
                          assignments, fixed)

    # reassign to model that minime log probability
    fixed_assignments = assignments[np.where(fixed)[0]]
    new_assignemnts = np.argmax(scores, axis=1)
    new_assignemnts[np.where(fixed)[0]] = fixed_assignments
    return new_assignemnts


def score_matrix(models, noise_models, sequences, assignments, fixed):
    # calculate log probability of each sequence on each model
    # import pdb; pdb.set_trace()
    all_models = np.concatenate((models, noise_models))
    n_models = len(all_models)
    n_sequences = assignments.size
    scores = np.empty((n_sequences, n_models))

    for i in range(n_sequences):
        for j, model in enumerate(all_models):
            scores[i, j] = model.log_probability(sequences[i, :])

    return scores


def total_log_prob(models, noise_models, sequences, assignments):
    # calculate log probability of current models + assignments
    all_models = np.concatenate((models, noise_models))
    logsum = 0
    for i, model in enumerate(all_models):
        in_model = np.where(assignments == i)[0]
        logsum += model.summarize(sequences[in_model, :])

    return logsum


def df_to_sequence_list(df):
    """
    pomegranate treats columns as sequences
    returns a transposed matrix and a list of labels corrsponding to sequences
    """
    sequences = df.as_matrix()
    labels = df.index.values
    return sequences, labels


def init_gaussian_hmm(sequences, n_states, model_id):
    """
    insantiate a model with random parameters
    randomly generates start and transition matrices
    generates nomal distrobutions for each state from partition on sequences
    """

    # make random transition probability matrix
    # scale each row to sum to 1
    trans = np.random.ranf((n_states, n_states))
    for i in range(n_states):
        trans[i, :] = trans[i, :] / trans[i, :].sum()

    """
    # make distrobutions from random partitioning of data
    temp_assignments = np.random.permutation([i % n_states for i
                                             in range(sequences.shape[1])])
    dists = []
    for i in range(n_states):
        in_state = np.where(temp_assignments == i)[0]
        dists.append(NormalDistribution.from_samples(sequences[:, in_state]))
    """

    # make distrobutions from random subsets of timepoints
    x = int(math.ceil(sequences.shape[1] / float(n_states)))
    # x = math.min(3, x)

    dists = []
    for i in range(n_states):
        temp_assignment = np.random.choice(sequences.shape[1], x)
        dists.append(NormalDistribution.from_samples
                     (sequences[:, temp_assignment]))

    # random start probabilities
    # scale to sum to 1
    starts = np.random.ranf(n_states)
    starts = starts / sum(starts)

    model = HiddenMarkovModel.from_matrix(trans, dists, starts, name=model_id)
    return model
