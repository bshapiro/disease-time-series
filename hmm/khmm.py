from pomegranate import NormalDistribution, HiddenMarkovModel
from collections import defaultdict
import math
import time
import numpy as np
import os

THREADCOUNT = 12
SHOW_TRAINING = True
ALGORITHM = 'baum-welch'
FIT_EPS = 1e-3  # Convergence threshold of fit
FIT_ITER = 1e3  # Max iter of fit function


def cluster(models, sequences, assignments, labels, fixed, eps,
            max_it, odir='./'):
    # import pdb; pdb.set_trace()
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

    # show performance under noise model -- lower bound for performance
    noise_model = models['noise']
    noise_log_prob = noise_model.summarize(sequences)

    with open(filepath, 'a') as f:
        print >> f, 'Log Prob Lower Bound = ', noise_log_prob, '\n'

    # initial model training
    models, assignments = train(models, sequences, assignments, FIT_EPS,
                                FIT_ITER)

    # intial probability calculation
    curr_log_prob = total_log_prob(models, sequences,
                                   assignments)
    start_time = time.time()

    with open(filepath, 'a') as f:
        print >> f, 'Init', ', Log Prob = ', str(curr_log_prob)
        print >> f, 'Assignments = ', ' '.join(
            [str(len(assignments[key])) for key in assignments])
        print >> f, 'Time Elapsed = 00:00:00'

    # iterative model assignment
    iteration = 0
    delta = eps

    while iteration <= max_it:
        # store prior assignments to short-circuit convergence
        prior_assignments = assignments.copy()

        # assign to models
        assignments = assign(models, sequences, assignments, fixed)

        # train on assignments
        models, assignments = train(models, sequences, assignments, FIT_EPS,
                                    FIT_ITER)

        # report improvement
        detla, curr_log_prob = report(models=models,
                                      sequences=sequences,
                                      assignments=assignments,
                                      iteration=iteration,
                                      filepath=filepath,
                                      curr_log_prob=curr_log_prob,
                                      start_time=start_time)

        print 'Iteration: ', iteration, ', Delta = ', detla, \
              ', Log Prob = ', curr_log_prob

        if delta < eps or prior_assignments == assignments:
            with open(filepath, 'a') as f:
                print >> f, 'Local optimum found at iter: ', (iteration)
            break

        iteration += 1

    # write conlusive lines in report
    converged = (delta < eps) or (prior_assignments == assignments)
    if not converged:
        if (iteration > max_it):
            line = 'Iteration limit reached: ' + str(max_it)
        else:
            line = 'Something went wrong'

        with open(filepath, 'a') as f:
            f.write(str(line))

    # write json representations of models
    for model_id, model in models.iteritems():
        filepath = odir.split('/') + [model_id]
        filepath = '/'.join(filepath)
        with open(filepath, 'w') as f:
            f.write(model.to_json())

    # write cluster assignments
    filepath = odir.split('/') + ['assignments.txt']
    filepath = '/'.join(filepath)
    with open(filepath, 'w') as f:
        for name, model in models.iteritems():
            f.write(name)
            f.write('\n')
            f.write('\t'.join(labels[assignments[name]]))
            # f.write(str(labels[np.where(assignments == i)]))
            f.write('\n')

    return models, assignments, converged


def assign(models, sequences, assignments, fixed):
    new_assignments = {}
    for model_id, model in models.iteritems():
        new_assignments[model_id] = []

    for i, sequence in enumerate(sequences):
        best_model = ''
        best_score = -1e1000
        for model_id, model in models.iteritems():
            # if fixed, keep it in the same model
            if fixed[i]:
                if i in assignments[model_id]:
                    best_model = model_id
                    break
            # otherwise put where it bet fits
            score = model.log_probability(sequence)
            if not np.isnan(score):
                if score > best_score:
                    best_model = model_id
                    best_score = score
        new_assignments[best_model].append(i)

    for model_id, model in models.iteritems():
        in_model = new_assignments[model_id]
        for seq_index in in_model:
            if fixed[seq_index]:
                continue
            elif np.isnan(model.log_probability(sequences[seq_index, :])):
                new_assignments[model_id].remove(seq_index)
                new_assignments['noise'].append(seq_index)

    return new_assignments


def train(models, sequences, assignments, eps, max_iter):
    new_models = {}
    for model_id, model in models.iteritems():
        new_model = models[model_id].copy()

        in_model = assignments[model_id]

        if len(in_model) > 0:
            sequence_set = sequences[in_model, :]

            # model parameter inertia, inversely proportional to the
            # proportions of examples to states
            inertia = 1 - (float(len(in_model)) / model.state_count())
            inertia = max(0, inertia)
            inertia = 0

            improvement = new_model.fit(sequence_set.astype(float),
                                        verbose=SHOW_TRAINING,
                                        algorithm=ALGORITHM,
                                        stop_threshold=eps,
                                        max_iterations=max_iter,
                                        edge_inertia=inertia,
                                        distribution_inertia=inertia,
                                        n_jobs=THREADCOUNT)

            if np.isnan(new_model.summarize(sequence_set)):
                new_model = model

            """
            for seq_index in in_model:
                if np.isnan(new_model.log_probability(sequences[seq_index, :])):
                    assignments[model_id].remove(seq_index)
                    assignments['noise'].append(seq_index)
            """
        new_models[model_id] = new_model

    return new_models, assignments


def total_log_prob(models, sequences, assignments):
    logsum = 0
    for model_id, model in models.iteritems():
        sequence_set = sequences[assignments[model_id], :]
        print model_id
        print model.summarize(sequence_set)
        if sequence_set.size > 0:
            logsum += model.summarize(sequence_set)

    return logsum


def log_prob_breakdown(models, sequences, assignments):
    logsum = []
    for model_id, model in models.iteritems():
        sequence_set = sequences[assignments[model_id], :]
        if sequence_set.size > 0:
            logsum.append(model.summarize(sequence_set))

    return logsum


def report(models, sequences, assignments, iteration,
           filepath, curr_log_prob, start_time):
    new_log_prob = total_log_prob(models, sequences, assignments)
    delta = new_log_prob - curr_log_prob
    time_elapsed = time.strftime("%H:%M:%S",
                                 time.gmtime(time.time() - start_time))

    with open(filepath, 'a') as f:
        print >> f, 'Iter = ', str(iteration)
        print >> f, 'Delta = ', delta, ', Log Prob: ', new_log_prob
        print >> f, 'Assignments = ', ' '.join(
            [str(len(assignments[key])) for key in assignments])
        print >> f, 'Time Elapsed = ', time_elapsed

    return delta, new_log_prob


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
