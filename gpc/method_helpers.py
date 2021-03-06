from collections import Counter
from GP import fit_gp
from GPCluster import GPCluster
from helpers import *
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
import numpy as np
import operator
import gc


def generate_initial_clusters(data, data_name):
    """
    Generates the initial clusters based on config.
    """
    if config['init'] is 'kmeans':
        kmeans_model = KMeans(config['k'], max_iter=1000)
        labels = kmeans_model.fit_predict(data)
        centroids = kmeans_model.cluster_centers_
        counter = Counter()
        for label in labels:
            counter[label] += 1
        print counter
    elif config['init'] is 'myeloma_paper':
        upreg, downreg, stable, teup, tedown = load_myeloma_paper_clusters()
        upreg_mean = np.mean(data[upreg], 0)
        downreg_mean = np.mean(data[downreg], 0)
        stable_mean = np.mean(data[stable], 0)
        teup_mean = np.mean(data[teup], 0)
        tedown_mean = np.mean(data[tedown], 0)
        centroids = [upreg_mean, downreg_mean, stable_mean, teup_mean, tedown_mean]
        labels = generate_myeloma_paper_labels()

    print "Estimating initial clusters..."
    i = 0
    gp_clusters = {}
    for centroid in centroids:
        y = centroid
        x = range(0, data.shape[1])
        name = data_name + str(i)
        gp = fit_gp(y, x, name)
        # gp = fit_gp_with_priors([2, 4, 6, 1, 2, 3], [0, 1, 2, 0, 1, 2], str(i))
        gp_clusters[name] = GPCluster(gp, name)
        i += 1

    init_likelihood = likelihood_given_init_clusters(data, labels, gp_clusters, data_name)
    print "Initial likelihood:", init_likelihood

    return gp_clusters, labels, init_likelihood


def generate_myeloma_paper_labels():
    """
    Labels each sample based on membership in myeloma paper clusters.
    """
    genes = [item[0] for item in pd.read_csv(open('../data/myeloma/genes.csv'), header=None).values.tolist()]
    upreg = [item[0] for item in pd.read_csv(open('../data/myeloma/upreg.csv'), header=None).values.tolist()]
    downreg = [item[0] for item in pd.read_csv(open('../data/myeloma/downreg.csv'), header=None).values.tolist()]
    stable = [item[0] for item in pd.read_csv(open('../data/myeloma/stable.csv'), header=None).values.tolist()]
    tedown = [item[0] for item in pd.read_csv(open('../data/myeloma/tedown.csv'), header=None).values.tolist()]
    teup = [item[0] for item in pd.read_csv(open('../data/myeloma/teup.csv'), header=None).values.tolist()]
    labels = []
    for gene in genes:
        if gene in upreg:
            labels.append(0)
        elif gene in downreg:
            labels.append(1)
        elif gene in stable:
            labels.append(2)
        elif gene in tedown:
            labels.append(3)
        elif gene in teup:
            labels.append(4)
        else:
            labels.append(-1)  # if not in clusters
    return labels


def likelihood_given_init_clusters(data, labels, gp_clusters, data_name):
    """
    Calculate the likelihood of the data given the initial cluster choices.
    """
    total_likelihood = 0
    index = 0
    for sample in data:
        label = labels[index]
        if label == -1:
            max_likelihood = max([gp_cluster.likelihood(sample, range(len(sample)), index) for gp_cluster in gp_clusters.values() if gp_cluster.name.startswith(data_name)])
            total_likelihood += max_likelihood
        else:
            cluster_name = data_name + str(label)
            total_likelihood += gp_clusters[cluster_name].likelihood(sample, range(len(sample)), index)
        index += 1
    return total_likelihood


def likelihood_for_clusters(gp_clusters):
    """
    Calculate the likelihood of all the data assigned to their clusters.
    """
    total_likelihood = 0
    for cluster in gp_clusters:
        if len(cluster.samples) != 0:
            # x = range(len(cluster.samples[0]))
            # index = 0
            # for sample in cluster.samples:
            #     total_likelihood += cluster.likelihood(sample, x, index)
            #     index += 1
            total_y = np.reshape(np.asarray(cluster.samples), (len(cluster.samples) * len(cluster.samples[0]), 1))
            total_x = range(len(cluster.samples[0]) * len(cluster.samples))
            total_x = np.reshape(np.asarray(total_x), (len(total_x), 1))
            total_likelihood = sum(cluster.likelihood(total_y, total_x, 0))
    return total_likelihood


def differential_transform(data):
    data = data.T
    new_data = (data - np.reshape(data[0, :], (1, 5680)))[1:].T
    return new_data


def find_max_corr_clusters(gp_clusters1, gp_clusters2, domain):
    """
    Finds the clusters to link. Samples each cluster over the domain to approximate
    that cluster and then computes pearson correlation (for now) between sets of samples.
    """
    num_related = config['num_related']
    corr = {}
    domain = np.reshape(domain, (domain.shape[0], 1))
    for cluster1 in gp_clusters1.values():
        for cluster2 in gp_clusters2.values():
            if corr.get((cluster2.name, cluster1.name)) is not None:
                continue
            cluster1_mean = cluster1.gp.predict(domain)[0]
            cluster2_mean = cluster2.gp.predict(domain)[0]
            corr[(cluster1.name, cluster2.name)] = pearsonr(cluster1_mean, cluster2_mean)[0]
    sorted_x = sorted(corr.items(), key=operator.itemgetter(1), reverse=True)
    num_left = num_related
    pairs = []

    while num_left > 0:  # sort the samples but don't include repeats
        pair = sorted_x.pop(0)[0]
        pairs.append(pair)
        new_sorted_x = []
        for item in sorted_x:
            if pair[0] in item[0]:
                continue
            elif pair[1] in item[0]:
                continue
            else:
                new_sorted_x.append(item)
        num_left = num_left - 1
        sorted_x = new_sorted_x

    print "Pairs:", pairs
    for pair in pairs:  # add the actual links to the GP cluster objects
        gp_cluster1 = gp_clusters1[pair[0]]
        gp_cluster2 = gp_clusters2[pair[1]]
        gp_cluster1.add_link(gp_cluster2)
        gp_cluster2.add_link(gp_cluster1)
    return pair_dict(pairs)


@unpack_args
def m_step(cluster, iteration):
    if cluster.samples == []:
        return cluster
    cluster.reestimate(iteration)
    return cluster


@unpack_args
def e_step(sample, gp_clusters):
    i = int(sample[0])
    sample = np.reshape(sample[1:], (len(sample[1:]), 1))

    sample_likelihoods = []
    for cluster in gp_clusters:  # find max likelihood cluster
        likelihood = cluster.likelihood(sample, range(len(sample)), i)
        sample_likelihoods.append(likelihood)

    max_likelihood = max(sample_likelihoods)
    max_index = sample_likelihoods.index(max_likelihood)

    return i, max_index


def assign_labeled_sample(sample, memberships, gp_clusters):
    i = sample[0]
    sample = sample[1:]
    gp_clusters[memberships[i]].assign_sample(sample, i)
