from hmmlearn import hmm, utils
import numpy as np


def rand_transition_matrix(n, m):
    """
    n rows
    m cols
    rows standardized to sum to 1
    """
    out = [[np.random.rand() for i in range(n)] for i in range(m)]
    out = [[i / sum(a) for i in a] for a in out]
    return out


def rand_gaussian_params(n):
    out = [[np.random.normal(), np.random.rand()] for i in range(n)]
    return out


def df_to_sequence_list(df):
    """
    domain is an ghmm emmisionsDomain object
    df is a pandas dataframe, we use the indx to label sequences
    """
    sequences = np.empty(0)
    lengths = np.empty(0)
    labels = np.empty(0)
    for seq in df.index:
        labels = np.append(labels, seq)
        sequences = np.concatenate([sequences,
                                   df.loc[seq].as_matrix()])
        lengths = np.append(lengths, df.loc[seq].as_matrix().size)
    return sequences.reshape(-1, 1), lengths, labels

def arr_to_sequence_list(arr):
    sequences = np.empty(0)
    lengths = np.empty(0)
    for i in range(arr.shape[0]):
        sequences = np.concatenate([sequences, arr[i, :]])
        lengths = np.append(lengths, arr[i, :].size)
    return sequences.reshape(-1, 1), lengths.astype(int)


def cluster(models, noise_models, sequences, lengths, assignments, fixed, eps, max_iter):
    """
    models - are list of hmm objects
    noise_models - a list of noise models which are never retrained
    eps is the total log probability difference to declare convergence
    max_iter is the maximum number of iterations to perform
    """
    curr_log_prob = -1e1000
    diff = eps
    iteration = 1
    print 'Iter = ', 0, ', Log Prob = ', curr_log_prob
    while iteration <= max_iter and diff >= eps:
        train(models, sequences, lengths, assignments)
        assignments = assign(models, noise_models, sequences, lengths,
                             assignments, fixed)
        new_log_prob = total_log_prob(models, noise_models, sequences, lengths,
                                      assignments)

        diff = new_log_prob - curr_log_prob
        curr_log_prob = new_log_prob
        print 'Iter = ', iteration, ', Delta Log Prob = ', diff, ', Log Prob = ', curr_log_prob
        iteration += 1

    converged = (diff < eps)
    return models, assignments, converged


def train(models, sequences, lengths, assignments):

    # train the models based on current assignment
    for i, model in enumerate(models):
        in_model = np.where(assignments == i)[0]
        if in_model.size != 0:
            sequence_set = sequences[get_seq_indices(lengths, in_model)]
            sequence_lengths = lengths[in_model]
            model.fit(sequence_set, sequence_lengths)


def assign(models, noise_models, sequences, lengths, assignments, fixed):
    scores = score_matrix(models, noise_models, sequences, lengths, assignments, fixed)

    # reassign to model that minimize log probability
    new_assignemnts = np.argmax(scores, axis=1)
    new_assignemnts[np.where(fixed)[0]] = fixed[np.where(fixed)[0]]
    return new_assignemnts

def score_matrix(models, noise_models, sequences, lengths, assignments, fixed):
    # calculate log probability of each sequence on each model
    # import pdb; pdb.set_trace()
    all_models = np.concatenate((models, noise_models))
    n_models = len(all_models)
    n_sequences = lengths.size
    scores = np.empty((n_sequences, n_models))

    for i in range(lengths.size):
        seq_indices = get_seq_indices(lengths, [i])
        for j, model in enumerate(all_models):
            scores[i, j] = model.score(sequences[seq_indices], [lengths[i]])

    return scores

def total_log_prob(models, noise_models, sequences, lengths, assignments):
    # calculate log probability of current models + assignments
    all_models = np.concatenate((models, noise_models))
    total = 0
    for i, model in enumerate(all_models):
        in_model = np.where(assignments == i)[0]
        if in_model.size != 0:
            sequence_set = sequences[get_seq_indices(lengths, in_model)]
            sequence_lengths = lengths[in_model]
            total += model.score(sequence_set, sequence_lengths)
    return total

def get_seq_indices(lengths, in_set):
    seq_indices = np.empty(0)
    for i in in_set:
        seq_indices = np.concatenate((seq_indices,
                                      np.arange(lengths[:i].sum(), lengths[:(i+1)].sum())
                                      ))
    return seq_indices.astype(int)
