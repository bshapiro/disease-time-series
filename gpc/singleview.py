###### MUST COME FIRST ######
import matplotlib as mpl
mpl.use('Agg')
#############################

from config import config
from helpers import *
from method_helpers import *
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cPickle import dump
from multiprocessing import Pool


def run_em(data, gp_clusters, labels):

    memberships = {}
    iterations = 0
    likelihoods = []

    for iteration in range(100):  # max 100 iterations, but we never hit this number

        for cluster in gp_clusters:  # unassign any samples assigned to clusters
            cluster.clear_samples()

        print "Running iteration ", iteration

        reassigned_samples = 0
        if config['parallel']:
            data_labeled = np.column_stack((range(data.shape[0]), data))
            pool = Pool()
            new_memberships = dict(pool.map(e_step, zip(data_labeled, [gp_clusters]*data.shape[0])))
            pool.close()
            pool.join()
            for key, value in new_memberships:
                if new_memberships[key] != memberships[key]:
                    reassigned_samples += 1
            memberships = new_memberships
        else:
            for i in range(data.shape[0]):  # iterate through samples

                sample = data[i]
                sample = np.reshape(sample, (len(sample), 1))

                sample_likelihoods = []
                for cluster in gp_clusters:  # find max likelihood cluster
                    likelihood = cluster.likelihood(sample, range(len(sample)), i)
                    sample_likelihoods.append(likelihood)

                max_likelihood = max(sample_likelihoods)
                max_index = sample_likelihoods.index(max_likelihood)

                gp_clusters[max_index].assign_sample(sample, i)  # assign samples to clusters
                if memberships.get(i) != max_index:
                    reassigned_samples += 1  # keep track of how many are reassigned
                    memberships[i] = max_index

        # test convergence
        print "Reassigned samples: ", reassigned_samples
        if reassigned_samples < 0.05*data.shape[0]:
            break

        e_likelihood = likelihood_for_clusters(gp_clusters)
        print "Likelihood after E step:", e_likelihood
        likelihoods.append(e_likelihood)

        if config['parallel']:
            pool = Pool()
            pool.map(m_step, zip(gp_clusters, [iteration]*len(gp_clusters)))
            pool.close()
            pool.join()
        else:
            for cluster in gp_clusters:  # reestimate all of the clusters
                if cluster.samples == []:
                    continue
                cluster.reestimate(iteration)

        m_likelihood = likelihood_for_clusters(gp_clusters)
        print "Likelihood after M step:", m_likelihood
        likelihoods.append(m_likelihood)

        iterations += 1

    print "Converged in ", iterations, " iterations."
    print "Number of reassigned samples in last iteration: ", reassigned_samples

    return gp_clusters, memberships, likelihoods

if __name__ == "__main__":
    config['views'] = 'single'
    polya = np.log2(pd.read_csv(open('../data/myeloma/polya.csv'), sep=',', header=None).as_matrix())
    ribosome = np.log2(pd.read_csv(open('../data/myeloma/ribosome.csv'), sep=',', header=None).as_matrix())
    te = load_te()
    datasets = {'polya': polya, 'ribosome': ribosome, 'te': te}
    data = datasets[config['dataset']]
    data = data[:500]  # TODO: REMOVE
    print "Shape:", data.shape

    if config['differential_transform']:
        print "Taking differential transform."
        data = differential_transform(data)
        print "Shape:", data.shape
        data = scale(data.T, with_mean=True, with_std=True).T

    else:
        data = scale(data.T, with_mean=True, with_std=True).T

    gp_clusters, labels, init_likelihood = generate_initial_clusters(data, config['dataset'])
    gp_clusters, memberships, likelihoods = run_em(data, gp_clusters.values(), labels)
    likelihoods.insert(0, init_likelihood)

    for cluster in gp_clusters:  # plot the final clusters with their background samples
        if cluster.samples == []:
            continue
        cluster.gp.plot()
        for sample in cluster.samples:
            plt.plot(sample, alpha=0.01)
        plt.savefig(generate_output_dir() + cluster.name + '_final_m.png')
        plt.close()
        plt.clf()

    plt.plot(likelihoods)
    plt.savefig(generate_output_dir() + '_likelihoods.png')
    plt.clf()

    dump(memberships, open(generate_output_dir() + 'memberships.dump', 'w'))  # dump memberships for further analysis
