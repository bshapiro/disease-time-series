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


def init(gene_list_file):
    subset = load(open(gene_list_file, 'r'))
    gc, mt, track = load_data()
    gc.data = gc.data.loc[subset, :]

    msequences, mlabels = df_to_sequence_list(mt.data)
    gsequences, glabels = df_to_sequence_list(gc.data)

    sequences = np.concatenate((msequences, gsequences), 0)
    labels = np.concatenate((mlabels, glabels))

    return sequences, labels


def gen_model(sequences, labels, algorithm, initialization, restarts, n, k,
              out_dir, base_id):

    if initialization == 'rand':
        init_method = init_gaussian_hmm
        init_args = {'n_states': n[0]}
    if init_lr_hmm == 'lr':
        init_method = init_lr_hmm
        s = n[0]
        sps = n[1]
        init_args = {'steps': s, 'state_per_step': sps, 'force_end': True}
    if initialization == 'cycle':
        init_method = init_cycle_hmm
        s = n[0]
        sps = n[1]
        init_args = {'steps': s, 'state_per_step': sps}

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
            in_model = assignments[model_id]
        print assignments
    # gen model for number of restarts
    for x in range(restarts):
        try:
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

            # add noise model
            models['noise'] = noise
            assignments['noise'] = []

            # perform clustering
            models, assignments, c = cluster(models=models,
                                             sequences=sequences,
                                             assignments=assignments,
                                             algorithm=algorithm,
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

        except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            print >> f, "Unexpected error:", sys.exc_info()[0]
            f.close()

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
    parsr.add_option("-i", "--initialization", dest="initialization",
                     default='rand', help="'rand', 'cycle', or 'lr'")
    parsr.add_option("-k", dest="k", default=None, type=int,
                     help="number of clusters")
    parsr.add_option("-n", dest="n", default=None, action="append", type=int,
                     help="states per cluster")
    parsr.add_option("-o", "--out_dir", dest="out_dir", default=None,
                     help="base directory to save output")
    parsr.add_option("-b", "--base_id", dest="base_id", default=None,
                     help="base name for output files")

    (options, args) = parsr.parse_args(sys.argv)

    sequences, labels = init(options.genefile)
    gen_model(sequences=sequences, labels=labels, algorithm=options.algorithm,
              initialization=options.initialization, restarts=options.restarts,
              n=options.n, k=options.k, out_dir=options.out_dir,
              base_id=options.base_id)
