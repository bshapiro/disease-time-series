import addpath
import pandas as pd
import numpy as np
from pomegranate import HiddenMarkovModel
from load_data import load_data
from pickle import load, dump
from scipy.cluster import hierarchy


def sequence_state_similarity(model, sequences):
    """
    computes a g * g matrix that gives the probability that two sequences
    were generated from the same sequence of hidden states
    """
    p = pd.DataFrame()
    for i, sequence1 in enumerate(sequences):
        seq1prob = model.predict_proba(sequence1)
        for j, sequence2 in enumerate(sequences):
            seq2prob = model.predict_proba(sequence2)
            s1s2logprob = np.sum(np.log(np.diag(seq1prob.dot(seq2prob.T))))
            p.iloc[i, j] = s1s2logprob
    return p


def joint_sequence_liklihood(sequence1, sequence2, model):
    """
    given two sequences and a model gives the log probability that both
    sequences were generated from the same sequence of hidden states
    """
    seq1prob = model.predict_proba(sequence1)
    seq2prob = model.predict_proba(sequence2)
    joint_log_prob = np.sum(np.log(np.diag(seq1prob.dot(seq2prob.T))))
    return joint_log_prob


def similarity(sequence_probs, clusters, model):
    """
    calculates joing cluster liklihood for all pairs of existing clusters
    returns a condensed matrix of average negative log probabilities
    """
    d = {i: key for i, key in enumerate(sorted(clusters.keys()))}
    similarity = np.empty((len(clusters), len(clusters)))
    similarity.fill(1e100)
    for i, c1 in enumerate(sorted(clusters.keys())):
        for j, c2 in enumerate(sorted(clusters.keys())):
            if j > i:
                avgll = joint_cluster_liklihood(sequence_probs, clusters[c1],
                                                clusters[c2], model)
                avgll = avgll / (len(clusters[c1]) + len(clusters[c2]))
                print i, j
                similarity[i, j] = avgll
    return similarity, d


def joint_cluster_liklihood(sequence_probs, cluster1, cluster2, model):
    """
    sequences is the matrix of sequnece data
    cluster1 and cluster2 are lists of indexes into sequences
    model is the hmm
    """
    cluster = list(set(cluster1 + cluster2))
    probmat = np.ones((data.columns.size, model.state_count() - 2))
    for sequence in data.loc[cluster, :].T:
        seqprob = sequence_probs[sequence]
        probmat = np.multiply(probmat, seqprob)
    p = probmat.sum(1)  # marginalize out states
    p = np.log(p)
    p = p.sum() * -1  # negative log prob
    return p


def linkage(data, model, start_clusters=None):
    """
    perform heirarchecal clustering on sequences over an HMM
    returns a linkage matrix Z formatted to scipy standards
    """
    Z = np.empty((1, 4))
    n = data.shape[0]
    if start_clusters is None:
        clusters = {i: [data.index[i]] for i in range(0, n)}
    else:
        clusters = start_clusters

    step = 0
    sequence_probs = {seq: model.predict_proba(data.loc[seq, :])
                      for seq in data.index.values}
    while len(clusters) > 1:
        sim, d = similarity(sequence_probs, clusters, model)
        # get index into clusters to be merged
        [i, j] = np.where(sim == sim.min())
        i = d[i[0]]
        j = d[j[0]]
        # new cluster id
        k = n + step
        # size of merged cluster
        size = len(clusters[i]) + len(clusters[j])
        # add new cluster, remove old
        clusters[k] = clusters.pop(i) + clusters.pop(j)
        # create linkage column and append it to z
        col = np.array([i, j, k, size])
        col = col.reshape(1, 4)
        Z = np.concatenate((Z, col), 0)
        step += 1
    Z = Z[1:, :]
    return Z


genefile = '1k_genes.p'
gc, mt, track = load_data()
genes = load(open(genefile,  'r'))
data = pd.concat((gc.data.loc[genes, :], mt.data))
sequences = data.as_matrix()

model_path = '../results/profile_hmm/1k5ssp/0/model'
model = HiddenMarkovModel.from_json(model_path)

# k = 100
Z = linkage(data, model)
dump(Z, open('linkage1k.p', 'wb'))
Z = hierarchy.linkage()
