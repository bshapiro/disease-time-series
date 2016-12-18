from config import config
from helpers import *
from method_helpers import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pickle


def run_em(datasets, gp_clusterings, related_pairs):

    iterations = 0
    memberships = {}
    for dataset_name in datasets.keys():
        memberships[dataset_name] = {}

    list_of_clusters = []
    for clustering in gp_clusterings.values():
        for cluster in clustering.values():
            list_of_clusters.append(cluster)

    for iteration in range(100):

        for cluster in list_of_clusters:
            cluster.clear_samples()

        reassigned_samples = 0

        print "Running iteration ", iteration
        for dataset_name, data in datasets.items():

            # if converged.get(dataset_name) is not None:
            #    continue

            gp_clusters = gp_clusterings[dataset_name]

            for j in range(data.shape[0]):
                sample = np.reshape(data[j], (len(data[j]), 1))

                max_likelihood = None
                max_cluster_name = None
                for cluster_name, cluster in gp_clusters.items():
                    likelihood = cluster.likelihood(sample, range(len(sample)), j)

                    # if cluster_name in related_pairs.values():
                    #     related_cluster_name = related_pairs[cluster_name]
                    #     related_dataset_name = related_cluster_name[:-1]
                    #     if memberships[related_dataset_name].get(j) is related_cluster_name:
                    #         likelihood = likelihood * (1.0 - config['strength'])

                    if max_likelihood is None:
                        max_likelihood = likelihood
                        max_cluster_name = cluster_name
                    elif likelihood > max_likelihood:
                        max_likelihood = likelihood
                        max_cluster_name = cluster_name

                gp_clusters[max_cluster_name].assign_sample(sample, j)

                if memberships[dataset_name].get(j) != max_cluster_name:
                    reassigned_samples += 1
                    memberships[dataset_name][j] = max_cluster_name

        print "Reassigned samples: ", reassigned_samples
        if reassigned_samples < 2*0.05*data.shape[0]:  # converged
            break

        print "Likelihood for E step:", likelihood_for_clusters(list_of_clusters)

        for dataset_name, data in datasets.items():
            for cluster in gp_clusterings[dataset_name].values():
                if cluster.samples == []:
                    continue
                cluster.reestimate(iteration)

        print "Likelihood after M step:", likelihood_for_clusters(list_of_clusters)

        iterations += 1

    print "Converged in ", iterations, " iterations."
    print "Number of reassigned samples in last iteration: ", reassigned_samples

    return gp_clusterings, memberships

if __name__ == "__main__":
    config['views'] = 'two'
    polya = np.log2(pd.read_csv(open('../data/myeloma/polya.csv'), sep=',', header=None).as_matrix())
    ribosome = np.log2(pd.read_csv(open('../data/myeloma/ribosome.csv'), sep=',', header=None).as_matrix())
    te = load_te()
    data1 = polya
    data2 = ribosome

    print "Shape 1:", data1.shape
    print "Shape 2:", data2.shape

    if config['differential_transform']:
        print "Taking differential transform."
        data1 = differential_transform(data1)
        data2 = differential_transform(data2)
        print "Shape 1:", data1.shape
        print "Shape 2:", data2.shape
        data1 = scale(data1.T, with_mean=False, with_std=True).T
        data2 = scale(data2.T, with_mean=False, with_std=True).T
    else:

        data1 = scale(data1.T, with_mean=True, with_std=True).T
        data2 = scale(data2.T, with_mean=True, with_std=True).T

    num_timesteps = data1.shape[1]

    gp_clusters1, labels1 = generate_initial_clusters(data1, 'polya')
    gp_clusters2, labels2 = generate_initial_clusters(data2, 'ribosome')
    related_pairs = find_max_corr_clusters(gp_clusters1, gp_clusters2, np.arange(0, num_timesteps - 1, 0.2))

    gp_clusterings = {'polya': gp_clusters1, 'ribosome': gp_clusters2}
    datasets = {'polya': data1, 'ribosome': data2}
    gp_clusterings, memberships = run_em(datasets, gp_clusterings, related_pairs)

    for clustering in gp_clusterings.values():
        for cluster in clustering.values():
            cluster.gp.plot()
            for sample in cluster.samples:
                plt.plot(sample, alpha=0.01)
            plt.savefig(generate_output_dir() + cluster.name + '_final_m.png')
            plt.close()
            plt.clf()

    pickle.dump(memberships, open(generate_output_dir() + 'memberships.dump', 'w'))
