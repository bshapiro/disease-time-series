from pomegranate import NormalDistribution, HiddenMarkovModel, State
import math
import time
import numpy as np
import os

THREADCOUNT = 12
SHOW_TRAINING = True
ALGORITHM = 'baum-welch'
FIT_EPS = 1e-3  # Convergence threshold of fit
FIT_ITER = 1e3  # Max iter of fit function


def cluster(models, sequences, assignments, labels, state_labels=None,
            fixed=None, tied=None, eps=1e-3, max_it=1e3, algorithm=ALGORITHM,
            odir='./'):

    # if fixed is not specified nothing is fixed
    if fixed is None:
        fixed = {}
        for model_id, model in models.iteritems():
            fixed[model_id] = []

    # if tied is not specified, no sequences are tied
    # make each sequence is its own group
    if tied is None:
        tied = {}
        for i, label in enumerate(labels):
            tied[label] = [i]

    # write initial cluster assignments
    filepath = odir.split('/') + ['init_assignments.txt']
    filepath = '/'.join(filepath)
    try:
        f = open(filepath, 'w')
    except:
        directory = '/'.join(filepath.split('/')[:-1])
        print "Creating directory...", directory
        os.makedirs(directory)
        f = open(filepath, 'w')
    f.close()

    with open(filepath, 'w') as f:
        for model_id, model in models.iteritems():
            f.write(model_id)
            f.write('\n')
            f.write('\t'.join(labels[assignments[model_id]]))
            f.write('\n')
            if state_labels is not None:
                f.write('\t'.join(state_labels[assignments[model_id]]))
            else:
                f.write('\n')
            # f.write(str(labels[np.where(assignments == i)]))
            f.write('\n')

    # for writing iteration report file
    filepath = odir.split('/') + ['iteration_report.txt']
    filepath = '/'.join(filepath)

    # show performance under noise model -- lower bound for performance
    noise_model = models['noise']
    noise_log_prob = noise_model.summarize(sequences)

    with open(filepath, 'a') as f:
        print >> f, 'Log Prob Lower Bound = ', noise_log_prob, '\n'

    # initial model training
    models, assignments = train(models, sequences, assignments, 'viterbi',
                                FIT_EPS, FIT_ITER)

    # intial probability calculation
    curr_log_prob = total_log_prob(models, sequences, assignments)
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
        assignments = assign(models, sequences, assignments, fixed, tied)

        # train on assignments
        models, assignments = train(models, sequences, assignments, algorithm,
                                    FIT_EPS, FIT_ITER)

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

        if (delta < 0) or (delta < eps):
            with open(filepath, 'a') as f:
                print >> f, 'Convergence threshold reached.', \
                    ' Local optimum found at iter: ', (iteration)
            break

        if (prior_assignments == assignments):
            with open(filepath, 'a') as f:
                print >> f, 'Stable cluster assignments.' \
                    ' Local optimum found at iter: ', (iteration)
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
        for model_id, model in models.iteritems():
            f.write(model_id)
            f.write('\n')
            f.write('\t'.join(labels[assignments[model_id]]))
            f.write('\n')
            if state_labels is not None:
                f.write('\t'.join(state_labels[assignments[model_id]]))
            else:
                f.write('\n')
            # f.write(str(labels[np.where(assignments == i)]))
            f.write('\n')

    return models, assignments, converged


def assign(models, sequences, assignments, fixed, tied):
    """
    model is a dictionary {'model_id': model}
    sequences is a matrix gene/metabolite x time
    assignments is a dictionary {'model_id' : row index to sequences}
    tied is a dictionary {group_id: tied seq indiced}
    fixed is a dictionary {'model_id': fixed group_id}
    one can be assigned
    """
    new_assignments = {}
    for model_id, model in models.iteritems():
        new_assignments[model_id] = []

    # one sequence from each group gets assigned
    for group_id, group_indices in tied.iteritems():
        # for each model evaluate group members
        best_in_group = -1
        best_model = ''
        best_score = -1e1000
        for model_id, model in models.iteritems():
            # find the best group member for the model
            best_in_group_in_model = -1
            best_score_in_model = -1e1000
            for i, sequence in enumerate(sequences[group_indices, :]):
                score = model.log_probability(sequence)
                if not np.isnan(score):
                    if score > best_score_in_model:
                        best_in_group_in_model = i
                        best_score_in_model = score
            # if fixed, keep it in the same model
            if group_id in fixed[model_id]:
                best_model = model_id
                best_in_group = best_in_group_in_model
                best_score = best_score_in_model
                assert(not np.isnan(best_score) and best_score > -1e1000)
                break
            # otherwise update best_model and best_in_group trackers
            if best_score_in_model > best_score \
               and best_in_group_in_model != -1:
                best_model = model_id
                best_in_group = best_in_group_in_model
                best_score = best_score_in_model

        if np.isnan(best_score) or best_score <= -1e1000:
            best_model = 'noise'
        new_assignments[best_model].append(tied[group_id][best_in_group])

    return new_assignments


def train(models, sequences, assignments, algorithm, eps, max_iter):
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

            new_model.fit(sequence_set.astype(float),
                          verbose=SHOW_TRAINING,
                          algorithm=algorithm,
                          stop_threshold=eps,
                          max_iterations=max_iter,
                          edge_inertia=inertia,
                          distribution_inertia=inertia,
                          n_jobs=THREADCOUNT)

            if np.isnan(new_model.summarize(sequence_set)):
                new_model = model

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
    """
    # make random transition probability matrix
    # scale each row to sum to 1
    trans = np.random.ranf((n_states, n_states))
    for i in range(n_states):
        trans[i, :] = trans[i, :] / trans[i, :].sum()

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
    """
    model = HiddenMarkovModel(model_id)

    # make states with distrobutions from random subsets of timepoints
    x = int(math.ceil(sequences.shape[1] / float(n_states)))
    states = []
    for i in range(n_states):
        temp_assignment = np.random.choice(sequences.shape[1], x)
        dist = \
            NormalDistribution.from_samples(sequences[:, temp_assignment])
        states.append(State(dist, name=str(i)))

    model.add_states(states)

    # add random start probabilities
    start_probs = np.random.ranf(n_states)
    start_probs = start_probs / start_probs.sum()
    for i, state in enumerate(states):
        model.add_transition(model.start, state, start_probs[i])

    # add random transition probabilites out of each state
    for state1 in states:
        transitions = np.random.ranf(n_states)
        transitions = transitions / transitions.sum()
        for i, state2 in enumerate(states):
            model.add_transition(state1, state2, transitions[i])

    model.bake()
    return model


def init_lr_hmm(sequences, steps, states_per_step,
                hard_end, model_id):
    """
    insantiate a left-right model with random parameters
    randomly generates start and transition matrices
    generates nomal distrobutions for each state from partition on sequences
    """
    model = HiddenMarkovModel(model_id)
    n_states = steps * states_per_step

    # make distrobutions from chronological subsets of timepoints
    step_size = int(math.ceil(sequences.shape[1] / float(n_states+1)))

    # generate states
    states = np.empty((steps, states_per_step), dtype=object)
    for i in range(steps):
        for j in range(states_per_step):
            temp_assignment = np.arange(step_size * i, step_size * (i+1))
            dist = \
                NormalDistribution.from_samples(sequences[:, temp_assignment])
            state_name = str(i) + '-' + str(j)
            states[i, j] = State(dist, name=str(state_name))

    # add states to model
    model.add_states(states.flatten().tolist())

    # make random transition from start -> step0
    trans = np.random.ranf(states_per_step)
    trans = trans / trans.sum()
    for j in range(states_per_step):
        model.add_transition(model.start, states[0, j], trans[j])

    # make random transition from step(i) -> step(i+1)
    for i in range(steps-1):
        for j in range(states_per_step):
            trans = np.random.ranf(states_per_step + 1)
            trans = trans / trans.sum()
            # self transition
            model.add_transition(states[i, j], states[i, j], trans[0])
            # out transition
            for x in range(states_per_step):
                model.add_transition(states[i, j], states[i + 1, x],
                                     trans[x + 1])

    # make random transition from stepn -> end
    if hard_end:
        for j in range(states_per_step):
            trans = np.random.ranf(2)
            trans = trans / trans.sum()
            # self transition
            model.add_transition(states[(steps - 1), j],
                                 states[(steps - 1), j], trans[0])
            # end transition
            model.add_transition(states[(steps - 1), j], model.end, trans[1])

    model.bake()
    return model


def init_cycle_hmm(sequences, steps, states_per_step, model_id):
    """
    insantiate a left-right model with random parameters
    randomly generates start and transition matrices
    generates nomal distrobutions for each state from partition on sequences
    """
    model = HiddenMarkovModel(model_id)
    n_states = steps * states_per_step

    # make distrobutions from chronological subsets of timepoints
    step_size = int(math.ceil(sequences.shape[1] / float(n_states+1)))

    # generate states
    states = np.empty((steps, states_per_step), dtype=object)
    for i in range(steps):
        for j in range(states_per_step):
            temp_assignment = np.arange(step_size * i, step_size * (i+1))
            dist = \
                NormalDistribution.from_samples(sequences[:, temp_assignment])
            state_name = str(i) + '-' + str(j)
            states[i, j] = State(dist, name=str(state_name))

    # add states to model
    model.add_states(states.flatten().tolist())

    # make random transition from start -> step0
    trans = np.random.ranf(n_states)
    trans = trans / trans.sum()
    for i, state in enumerate(states.flatten().tolist()):
        model.add_transition(model.start, state, trans[i])

    # make random transition from step(i) -> step(i+1)
    for i in range(steps-1):
        for j in range(states_per_step):
            trans = np.random.ranf(states_per_step + 1)
            trans = trans / trans.sum()
            # self transition
            model.add_transition(states[i, j], states[i, j], trans[0])
            # out transition
            for x in range(states_per_step):
                model.add_transition(states[i, j], states[i + 1, x],
                                     trans[x + 1])

    # make random transition from stepn -> step0
    for j in range(states_per_step):
        trans = np.random.ranf(states_per_step + 1)
        trans = trans / trans.sum()
        # self transition
        model.add_transition(states[(steps - 1), j], states[(steps - 1), j],
                             trans[0])
        # out transition
        for x in range(states_per_step):
            model.add_transition(states[(steps - 1), j], states[0, x],
                                 trans[x + 1])
    model.bake()
    return model
