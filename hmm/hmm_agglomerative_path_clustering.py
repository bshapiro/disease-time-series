import addpath
import pandas as pd
import numpy as np
import sys
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
    p = p.sum()  # negative log prob
    return p


def gen_mergemat_complete(sequence_probs, clusters, model):
    """
    calculates joing cluster liklihood for all pairs of existing clusters
    returns a condensed matrix of average negative log probabilities
    """
    mergemat = np.empty((len(clusters), len(clusters)))
    mergemat.fill(-1 * np.inf)
    print "Generating merged liklihoods..."
    for i, c1 in enumerate(sorted(clusters.keys())):
        for j, c2 in enumerate(sorted(clusters.keys())):
            if j > i:
                avgll = joint_cluster_liklihood(sequence_probs, clusters[c1],
                                                clusters[c2], model)
                avgll = avgll / (len(clusters[c1]) + len(clusters[c2]))
                print i, j
                mergemat[i, j] = avgll
                mergemat[j, i] = avgll

    mergemat = pd.DataFrame(data=mergemat, index=sorted(clusters.keys()),
                            columns=sorted(clusters.keys()))
    return mergemat


def make_merge(clusters, mergemat, k, sequence_probs, model):
    """
    Performs cluster merge
    clusters are the current clusters
    d is a dictionary mapping indices to cluster ids
    mergemat is the average liklihood of merging existing clusters
    k is the id of the new merged clusters
    similarity is the pairwise similarity matrix
    model is the HMM
    """

    # choose clusters to merge
    m = np.where(mergemat == mergemat.max().max())
    o1 = mergemat.index[m[0][0]]
    o2 = mergemat.index[m[1][0]]

    # merge clusters
    mm = mergemat.max().max()
    print 'Merging clusters:', o1, o2, mm, k
    if mm <= (-1 * np.inf):
        return None, clusters, mergemat

    clusters[k] = clusters.pop(o1) + clusters.pop(o2)
    ksize = len(clusters[k])

    # drop old clusters
    mergemat = mergemat.drop(o1, 0)
    mergemat = mergemat.drop(o1, 1)
    mergemat = mergemat.drop(o2, 0)
    mergemat = mergemat.drop(o2, 1)

    # add new cluster to mergemat
    mergemat.loc[k, :] = -1 * np.inf
    mergemat.loc[:, k] = -1 * np.inf
    for cid in mergemat.index:
        if cid != k:
            avgll = joint_cluster_liklihood(sequence_probs, clusters[cid],
                                            clusters[k], model)
            avgll = np.log(np.exp(avgll /
                                  (len(clusters[k]) + len(clusters[cid]))))

            mergemat.loc[cid, k] = avgll
            mergemat.loc[k, cid] = avgll

    # assemble column of linkage matrix
    col = np.array([o1, o2, -1 * mm, ksize])
    col = col.reshape(1, 4)
    return col, clusters, mergemat


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

    sequence_probs = {seq: model.predict_proba(data.loc[seq, :])
                      for seq in data.index.values}
    print len(sequence_probs)
    mergemat = gen_mergemat_complete(sequence_probs, clusters, model)

    # cluster merging
    k = n  # new cluster id
    while len(clusters) > 1:
        # perform merge
        col, clusters, mergemat = \
            make_merge(clusters, mergemat, k, sequence_probs, model)
        if col is None:
            break

        Z = np.concatenate((Z, col), 0)
        k += 1

    q = k  # new cluster id
    while len(clusters) > 1:
        # perform fake merge
        print 'Fake merge:', q
        c1 = clusters.keys()[0]
        c2 = clusters.keys()[1]
        clusters[q] = clusters.pop(c1) + clusters.pop(c2)
        col = np.array([c1, c2, q, len(clusters[q])])
        col = col.reshape(1, 4)
        Z = np.concatenate((Z, col), 0)
        q += 1

    Z = Z[1:, :]
    return Z, k


def sequence_state_liklihood(sequence, statepath, model):
    tm = model.dense_transition_matrix()
    a = statepath[0]
    lp = tm[model.start_index, a]
    lp += model.states[a].distribution.log_probability(sequence[0])
    for i in range(1, len(statepath)):
        a = statepath[i - 1]
        b = statepath[i]
        lp += tm[a, b]  # transition log prob
        # log prob of observation
        lp += model.states[b].distribution.log_probability(sequence[i])
    return lp


def vit_cluster_liklihood(similarity, cluster1, cluster2, model):
    """
    sequences is the matrix of sequnece data
    cluster1 and cluster2 are lists of indexes into sequences
    model is the hmm
    """
    cluster = list(set(cluster1 + cluster2))
    joint_liklihood = np.empty(len(cluster))
    for i, seq1 in enumerate(data.loc[cluster, :].index):
        subsum = 0
        for j, seq2 in enumerate(data.loc[cluster, :].index):
            subsum += similarity.loc[seq1, seq2]
        subsum = np.exp(subsum)
        joint_liklihood[i] = subsum
    joint_liklihood = np.log(np.sum(joint_liklihood))
    return joint_liklihood


def viterbi_pair_similarity(data, vpaths, model):
    """
    calculates joing cluster liklihood for all pairs of existing clusters
    returns a condensed matrix of average negative log probabilities
    """
    similarity = np.empty((data.shape[0], data.shape[0]))
    for i, seq1 in enumerate(data.index):
        for j, seq2 in enumerate(data.index):
            similarity[i, j] = sequence_state_liklihood(data.loc[seq1, :],
                                                        vpaths[seq2], model)
            print i, j
    similarity = pd.DataFrame(data=similarity, index=data.index,
                              columns=data.index)
    return similarity


def gen_mergemat(similarity, clusters, model):
    """
    calculates joing cluster liklihood for all pairs of existing clusters
    returns a condensed matrix of average negative log probabilities
    """
    mergemat = np.empty((len(clusters), len(clusters)))
    mergemat.fill(-1 * np.inf)
    mergemat = pd.DataFrame(data=mergemat, index=clusters.keys(),
                            columns=clusters.keys())
    for i, c1 in enumerate(mergemat.index):
        for j, c2 in enumerate(mergemat.columns):
            if j > i:
                avgll = vit_cluster_liklihood(similarity, clusters[c1],
                                              clusters[c2], model)
                avgll = np.log(np.exp(avgll /
                                      (len(clusters[c1]) + len(clusters[c2]))))
                print i, j
                mergemat.loc[c1, c2] = avgll
                mergemat.loc[c2, c1] = avgll

    return mergemat


def vit_make_merge(clusters, mergemat, k, similarity, model):
    """
    Performs cluster merge
    clusters are the current clusters
    d is a dictionary mapping indices to cluster ids
    mergemat is the average liklihood of merging existing clusters
    k is the id of the new merged clusters
    similarity is the pairwise similarity matrix
    model is the HMM
    """

    # choose clusters to merge
    m = np.where(mergemat == mergemat.max().max())
    o1 = mergemat.index[m[0][0]]
    o2 = mergemat.index[m[1][0]]

    # merge clusters
    mm = mergemat.max().max()
    if mm <= (-1 * np.inf):
        return None, clusters, mergemat

    print 'Merging clusters:', o1, o2, mm
    clusters[k] = clusters.pop(o1) + clusters.pop(o2)
    ksize = len(clusters[k])

    # drop old clusters
    mergemat = mergemat.drop(o1, 0)
    mergemat = mergemat.drop(o1, 1)
    mergemat = mergemat.drop(o2, 0)
    mergemat = mergemat.drop(o2, 1)

    # add new cluster to mergemat
    mergemat.loc[k, :] = -1 * np.inf
    mergemat.loc[:, k] = -1 * np.inf
    for cid in mergemat.index:
        if cid != k:
            avgll = vit_cluster_liklihood(similarity, clusters[cid],
                                          clusters[k], model)
            avgll = np.log(np.exp(avgll /
                                  (len(clusters[cid]) + len(clusters[k]))))
            mergemat.loc[cid, k] = avgll
            mergemat.loc[cid, k] = avgll

    # assemble column of linkage matrix
    col = np.array([o1, o2, -1 * mm, ksize])
    col = col.reshape(1, 4)
    return col, clusters, mergemat


def fast_linkage(data, model, start_clusters=None):
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
    vpaths = {seq: model.predict(data.loc[seq, :])
              for seq in data.index.values}

    similarity = viterbi_pair_similarity(data, vpaths, model)
    mergemat = gen_mergemat(similarity, clusters, model)

    # cluster merging
    while len(clusters) > 1:
        k = n + step  # new cluster id
        # perform merge
        try:
            col, clusters, mergemat = \
                vit_make_merge(clusters, mergemat, k, similarity, model)

            Z = np.concatenate((Z, col), 0)
            step += 1
        except:
            break

    while len(clusters) > 1:
        q = n + step  # new cluster id
        # perform fake merge
        c1 = clusters.keys()[0]
        c2 = clusters.keys()[1]
        clusters[q] = clusters.pop(c1) + clusters.pop(c2)
        col = np.array([c1, c2, q, len(clusters[q])])
        col = col.reshape(1, 4)
        Z = np.concatenate((Z, col), 0)
        step += 1

    Z = Z[1:, :]
    return Z, mergemat, clusters, data, similarity

genefile = '1k_genes.p'
gc, mt, track = load_data()
genes = load(open(genefile,  'r'))
data = gc.data.loc[genes, :]
sequences = data.as_matrix()

model_path = '/'.join((sys.argv[1]).split('/') + ['model'])
model = HiddenMarkovModel.from_json(model_path)

Z, k = linkage(data, model)
linkage_path = '/'.join(model_path.split('/')[:-1] + ['linkage.p'])
dump(Z, open(linkage_path, 'wb'))

# k = 100
# Z, mm, c, d, s = fast_linkage(data.iloc[:50, :], model)
# linkage_path = '/'.join(model_path.split('/')[:-1] + ['linkage.p'])
# dump(Z, open(linkage_path, 'wb'))
