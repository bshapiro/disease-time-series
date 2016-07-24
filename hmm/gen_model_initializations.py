from optparse import OptionParser
import sys
import numpy as np

# from load_data import load_data
import time
from pomegranate import NormalDistribution, HiddenMarkovModel
from pickle import load
from load_data import load_data
from khmm import (df_to_sequence_list, cluster, init_gaussian_hmm, init_lr_hmm,
                  init_cycle_hmm, total_log_prob)


def init(gene_list_file, invert=False):
    subset = load(open(gene_list_file, 'r'))
    gc, mt, track = load_data()
    gc.data = gc.data.loc[subset, :]

    msequences, mlabels = df_to_sequence_list(mt.data)
    gsequences, glabels = df_to_sequence_list(gc.data)

    sequences = np.concatenate((msequences, gsequences), 0)
    sequences = np.concatenate((sequences, -1 * sequences))
    labels = np.concatenate((mlabels, glabels))
    tied = {}
    print 'Invert is: ', invert
    if invert:
        for i, label in enumerate(labels):
            tied[label] = [i, i+labels.size]
        labels = np.concatenate(((labels + '+'), (labels + '-')))
    else:
        for i, label in enumerate(labels):
            tied[label] = [i]

    return sequences, labels, tied


def gen_model(sequences, labels, algorithm, initialization, restarts, n, k,
              out_dir, base_id, tied, invert):

    if initialization == 'rand':
        init_method = init_gaussian_hmm
        init_args = {'n_states': n[0][0]}
    if initialization == 'lr':
        init_method = init_lr_hmm
        s = n[0][0]
        sps = n[0][1]
        curr = 0
        init_args = {'steps': s, 'states_per_step': sps,
                     'force_end': True}
    if initialization == 'cycle':
        init_method = init_cycle_hmm
        s = n[0][0]
        sps = n[0][1]
        curr = 0
        init_args = {'steps': s, 'states_per_step': sps}

    best = 0
    best_score = -1e1000

    # genes/metabolites will be assigned to noise model if other models
    # fail to model it better
    noise_dist = [NormalDistribution(0, 1)]
    noise_trans = np.array([[1]])
    starts = np.array([1])
    noise = HiddenMarkovModel.from_matrix(noise_trans, noise_dist, starts,
                                          name='noise')
    noise.freeze_distributions()

    np.random.seed(int(time.time()))
    randassigns = []
    for x in range(restarts):
        randassigns.append(np.random.randint(k, size=labels.size))

    for x in range(restarts):
        randassign = randassigns[x]
        assignments = {}
        for i in range(k):
            model_id = str(i)
            assignments[model_id] = \
                np.where(randassign == i)[0].tolist()

    # gen model for number of restarts
    for x in range(restarts):
        #try:
        collection_id = base_id + '_' + str(x)
        odir = '/'.join(out_dir.split('/') + [collection_id])

        print 'Learning: ', collection_id

        # generate random initial assignments
        # initialize models on random assignments
        randassign = randassigns[x]
        assignments = {}
        models = {}
        for i in range(k):
            model_id = str(i)
            assignments[model_id] = \
                np.where(randassign == i)[0].tolist()
            in_model = assignments[model_id]
            models[model_id] = \
                init_method(sequences[in_model, :], model_id=model_id,
                            **init_args)
            if initialization == 'cycle' or initialization == 'lr':
                curr += 1
                curr %= len(n)
                init_args['steps'] = n[curr][0]
                init_args['states_per_step'] = n[curr][1]
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
                                         algorithm=algorithm,
                                         fixed=fixed, tied=tied,
                                         labels=labels,
                                         odir=odir)

        score = total_log_prob(models, sequences, assignments)
        if best_score < score:
            best_score = score
            best = collection_id
            bestfile = '/'.join(out_dir.split('/') + ['best'])
            with open(bestfile, 'w') as f:
                print >> f, collection_id
                f.close()

        """except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            print >> f, "Unexpected error:", sys.exc_info()[0]
            f.close()"""

    return best


if __name__ == "__main__":
    parsr = OptionParser()
    # input options
    parsr.add_option('-g', "--genefile", dest="genefile", default='3k_genes.p',
                     help="pickle of np array of gene subset")
    parsr.add_option('-a', "--algorithm", dest="algorithm",
                     default='baum-welch',
                     help="'viterbi' or 'baum-welch'")
    parsr.add_option('-r', "--restarts", dest="restarts", type=int, default=1,
                     help="number of times to create model")
    parsr.add_option("--invert", dest="invert", action="store_true",
                     default=False,
                     help="whether to allow sequence inversion")
    parsr.add_option("-i", "--initialization", dest="initialization",
                     default='rand', help="'rand', 'cycle', or 'lr'")
    parsr.add_option("-k", dest="k", default=None, type=int,
                     help="number of clusters")
    parsr.add_option("-n", dest="n", default=None, type=int, action='append',
                     nargs=2, help="states per cluster")
    parsr.add_option("-o", "--out_dir", dest="out_dir", default='./',
                     help="base directory to save output")
    parsr.add_option("-b", "--base_id", dest="base_id", default="",
                     help="base name for output files")

    (options, args) = parsr.parse_args(sys.argv)

    sequences, labels, tied = init(options.genefile, invert=options.invert)
    gen_model(sequences=sequences, labels=labels, algorithm=options.algorithm,
              initialization=options.initialization, restarts=options.restarts,
              n=options.n, k=options.k, tied=tied, out_dir=options.out_dir,
              base_id=options.base_id, invert=options.invert)
