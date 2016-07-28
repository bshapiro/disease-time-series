import numpy as np
import math
from pomegranate import HiddenMarkovModel, State, NormalDistribution


def gaussian_hmm(sequences, n_states, model_id, seed=None):
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
    # seed random numer generator
    if seed is not None:
        np.random.seed(seed)

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
    print 'Initialized HMM: ', model.name
    return model


def lr_hmm(sequences, steps, states_per_step, self_trans=True, force_end=False,
           model_id='Left-Righ HMM', seed=None):
    """
    insantiate a left-right model with random parameters
    randomly generates start and transition matrices
    generates nomal distrobutions for each state from partition on sequences
    force_end if we require sequence to end in end state
    """

    # seed random number generator
    if seed is not None:
        np.random.seed(seed)

    model = HiddenMarkovModel(model_id)
    n_states = steps * states_per_step

    # make distrobutions from chronological subsets of timepoints
    step_size = int(math.ceil(sequences.shape[1] / float(n_states+1)))
    samples_per_step = int(math.ceil(sequences.shape[0] / states_per_step))

    # generate states
    states = np.empty((steps, states_per_step), dtype=object)
    for i in range(steps):
        for j in range(states_per_step):
            time_assignment = np.arange(step_size * i, step_size * (i+1))
            temp_assignment = np.random.choice(sequences.shape[0],
                                               samples_per_step, replace=False)

            dist = \
                NormalDistribution.from_samples(sequences[temp_assignment,
                                                time_assignment])
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
            # if allowing self transition, add self transition
            if self_trans:
                trans = trans / trans.sum()
                model.add_transition(states[i, j], states[i, j], trans[0])
            # otherwise ignore and renormalize transition probabilities
            else:
                trans[0] = 0
                trans = trans / trans.sum()
            # set out transitions
            for x in range(states_per_step):
                model.add_transition(states[i, j], states[i + 1, x],
                                     trans[x + 1])

    # make random transition from stepn -> end
    if force_end:
        for j in range(states_per_step):
            trans = np.random.ranf(2)
            trans = trans / trans.sum()
            # self transition
            model.add_transition(states[(steps - 1), j],
                                 states[(steps - 1), j], trans[0])
            # end transition
            model.add_transition(states[(steps - 1), j], model.end, trans[1])

    model.bake()
    print 'Initialized Left-Right HMM:', model.name, '[', \
        steps, states_per_step, '], Self Transistion: ', self_trans
    return model


def cycle_hmm(sequences, steps, states_per_step, self_trans=True,
              model_id='Cycle HMM', seed=None):
    """
    insantiate a left-right model with random parameters
    randomly generates start and transition matrices
    generates nomal distrobutions for each state from partition on sequences
    """

    # seed random number generator
    if seed is not None:
        np.random.seed(seed)

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
            # if allowing self transition, add self transition
            if self_trans:
                trans = trans / trans.sum()
                model.add_transition(states[i, j], states[i, j], trans[0])
            # otherwise ignore and renormalize transition probabilities
            else:
                trans[0] = 0
                trans = trans / trans.sum()
            # set out transitions
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
    print 'Initialized Cyclic State HMM:', '[', \
        steps, states_per_step, ']'
    return model
