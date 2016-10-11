import addpath
import pandas as pd
import numpy as np
import sys
from pomegranate import HiddenMarkovModel
from load_data import load_data
from pickle import load, dump
from scipy.cluster import hierarchy
from collections import defaultdict
from itertools import combinations
from heapq import heappush, heappop
import os


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


def cluster_liklihood(sequence_probs, cluster, model):
    """
    sequences is the matrix of sequnece data
    cluster1 and cluster2 are lists of indexes into sequences
    model is the hmm
    """
    probmat = np.ones((data.columns.size, model.state_count() - 2))
    for sequence in data.loc[cluster, :].T:
        seqprob = sequence_probs[sequence]
        probmat = np.multiply(probmat, seqprob)
    p = probmat.sum(1)  # marginalize out states
    p = np.log(p)
    p = p.sum()  # log prob
    return p


def joint_cluster_liklihood(sequence_probs, cluster1, cluster2, model):
    """
    sequences is the matrix of sequnece data
    cluster1 and cluster2 are lists of indexes into sequences
    model is the hmm
    """
    cluster = list(set(cluster1 + cluster2))
    return cluster_liklihood(sequence_probs, cluster, model)


def gen_mergemat_complete(sequence_probs, clusters, model):
    """
    calculates increase in negative log prob resulting from merge
    for all potential merges
    """
    mergemat = np.empty((len(clusters), len(clusters)))
    mergemat.fill(-1 * np.inf)
    cluster_probs = {}
    # fill in diagonal with current cluster log liklihoods
    print "Generating merged liklihoods..."
    for i, c1 in enumerate(sorted(clusters.keys())):
        cluster_probs[c1] = cluster_liklihood(sequence_probs, clusters[c1],
                                              model)
    # fill upper triangular entries log liklihood decrease due to
    for i, c1 in enumerate(sorted(clusters.keys())):
        for j, c2 in enumerate(sorted(clusters.keys())):
            if j > i:
                nlp = joint_cluster_liklihood(sequence_probs,
                                              clusters[c1],
                                              clusters[c2], model)
                print i, j
                mergemat[i, j] = nlp - cluster_probs[c1] - cluster_probs[c2]

    mergemat = pd.DataFrame(data=mergemat, index=sorted(clusters.keys()),
                            columns=sorted(clusters.keys()))
    return mergemat, cluster_probs


def make_merge(clusters, mergemat, cluster_probs, k, sequence_probs, model):
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
    """
    o1 = 0
    o2 = 1
    mm = mergemat.iloc[o1, o2]
    for i in range(0, mergemat.index.size):
        for j in range(i, mergemat.index.size):
            if (j > i) and (mm < mergemat.iloc[i, j]):
                o1 = i
                o2 = j
                mm = mergemat.iloc[i, j]

    o1 = mergemat.index[o1]
    o2 = mergemat.index[o2]
    """

    o1, o2 = np.where(mergemat == mergemat.max().max())
    o1 = mergemat.index[o1[0]]
    o2 = mergemat.columns[o2[0]]
    mm = mergemat.loc[o1, o2]

    print 'Merging clusters:', o1, o2, mm, k
    if mm <= (-1 * np.inf):
        return None, clusters, mergemat, cluster_probs

    clusters[k] = clusters.pop(o1) + clusters.pop(o2)
    cluster_probs.pop(o1)
    cluster_probs.pop(o2)
    cluster_probs[k] = cluster_liklihood(sequence_probs, clusters[k],
                                         model)
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
            nlp = joint_cluster_liklihood(sequence_probs, clusters[cid],
                                          clusters[k], model)

            mergemat.loc[cid, k] = \
                nlp - cluster_probs[k] - cluster_probs[cid]

    # assemble column of linkage matrix
    col = np.array([o1, o2, -1 * mm, ksize])
    col = col.reshape(1, 4)
    return col, clusters, mergemat, cluster_probs


def gen_mergeheap(sequence_probs, clusters, model):
    print "Generating merged liklihoods..."
    cluster_probs = {}
    for c1 in clusters.keys():
        cluster_probs[c1] = cluster_liklihood(sequence_probs, clusters[c1],
                                              model)
    mergeheap = []
    keys = sorted(clusters.keys())
    for i, cid in enumerate(keys):
        subheap = []
        for merge in [(cid, a) for a in keys[(i + 1):]]:
            print merge
            c1 = clusters[merge[0]]
            c2 = clusters[merge[1]]
            cost = joint_cluster_liklihood(sequence_probs, c1, c2, model) - \
                cluster_probs[merge[0]] - cluster_probs[merge[1]]
            cost *= -1
            heappush(subheap, (cost, merge))

        # only keep the best merge for that cid
        if len(subheap) > 0:
            heappush(mergeheap, heappop(subheap))

    """
    merges = combinations(clusters.keys(), 2)
    mergeheap = []
    for merge in merges:
        print merge
        c1 = clusters[merge[0]]
        c2 = clusters[merge[1]]
        cost = joint_cluster_liklihood(sequence_probs, c1, c2, model) - \
            cluster_probs[merge[0]] - cluster_probs[merge[1]]
        cost *= -1
        heappush(mergeheap, (cost, merge))
    """
    return mergeheap, cluster_probs


def shrink_heap(mergeheap, keys):
    lookup = defaultdict(bool)
    for key in keys:
        lookup[key] = True
    newheap = []
    while len(mergeheap) > 0:
        cost, merge = heappop(mergeheap)
        if (lookup[merge[0]] and lookup[merge[1]]):
            heappush(newheap, (cost, merge))
    return newheap


def merge(clusters, mergeheap, cluster_probs, k, sequence_probs, model):
    # pop off the heap until we have a valid merge or no merges possible
    cost, merge = heappop(mergeheap)
    while not (merge[0] in clusters and merge[1] in clusters):
        if len(mergeheap) > 0:
            # look at next best merge
            cost, merge = heappop(mergeheap)
        else:
            # no valid merges left
            return None

    clusters[k] = clusters.pop(merge[0]) + clusters.pop(merge[1])
    cluster_probs[k] = cluster_liklihood(sequence_probs, clusters[k], model)
    # push new merges onto the heap
    subheap = []
    for i in clusters.keys():
        if i != k:
            c1 = clusters[k]
            c2 = clusters[i]
            cost = joint_cluster_liklihood(sequence_probs, c1, c2, model) - \
                cluster_probs[k] - cluster_probs[i]
            cost *= -1
            heappush(subheap, (cost, (k, i)))
    if len(subheap) > 0:
        heappush(mergeheap, heappop(subheap))

    col = np.array([merge[0], merge[1], cost, len(clusters[k])]).reshape(1, 4)
    print 'Merged clusters:', merge[0], merge[1], cost, k, len(mergeheap)

    return col, mergeheap


def heap_linkage(data, model):
    Z = np.empty((1, 4))
    n = data.shape[0]
    clusters = {i: [data.index[i]] for i in range(0, n)}

    sequence_probs = {seq: model.predict_proba(data.loc[seq, :])
                      for seq in data.index.values}
    print len(sequence_probs)
    mergeheap, cluster_probs = gen_mergeheap(sequence_probs, clusters, model)

    # cluster merging
    k = n  # new cluster id
    while len(clusters) > 1:
        # perform merge
        col, mergeheap = merge(clusters, mergeheap, cluster_probs, k,
                               sequence_probs, model)
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
    mergemat, cluster_probs = \
        gen_mergemat_complete(sequence_probs, clusters, model)

    # cluster merging
    k = n  # new cluster id
    while len(clusters) > 1:
        # perform merge
        col, clusters, mergemat, cluster_probs = \
            make_merge(clusters, mergemat, cluster_probs, k, sequence_probs, model)
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


def vit_cluster_liklihood(vpaths, cluster, model):
    """
    sequences is the matrix of sequnece data
    cluster1 and cluster2 are lists of indexes into sequences
    model is the hmm
    """
    """
    joint_liklihood = np.empty(len(cluster))
    paths = defaultdict(list)
    for member, sequence in cluster.iteritems():
        for i in range(len(vpaths[member])):
            paths[i] = paths[i].append(vpaths[member][i])
    for i in range(1, len(paths)):


    for i, seq1 in enumerate(data.loc[cluster, :].index):
        subsum = 0
        for j, seq2 in enumerate(data.loc[cluster, :].index):
            subsum += similarity.loc[seq1, seq2]
        subsum = np.exp(subsum)
        joint_liklihood[i] = subsum
    joint_liklihood = np.log(np.sum(joint_liklihood))
    return joint_liklihood"""
    return None


def vit_joint_cluster_liklihood(vpaths, cluster1, cluster2, model):
    """
    sequences is the matrix of sequnece data
    cluster1 and cluster2 are lists of indexes into sequences
    model is the hmm
    """
    cluster = list(set(cluster1 + cluster2))
    return vit_cluster_liklihood(vpaths, cluster, model)


def gen_mergemat_vit(vpaths, clusters, model):
    """
    calculates joing cluster liklihood for all pairs of existing clusters
    returns a condensed matrix of average negative log probabilities
    """

    mergemat = np.empty((len(clusters), len(clusters)))
    mergemat.fill(-1 * np.inf)
    print "Generating merged liklihoods..."
    for i, c1 in enumerate(sorted(clusters.keys())):
        mergemat[i, i] = vit_cluster_liklihood(vpaths, clusters[c1],
                                               model)
    for i, c1 in enumerate(sorted(clusters.keys())):
        for j, c2 in enumerate(sorted(clusters.keys())):
            if j > i:
                nlp = vit_joint_cluster_liklihood(vpaths,
                                                  clusters[c1],
                                                  clusters[c2], model)
                print i, j
                mergemat[i, j] = nlp - mergemat[i, i] - mergemat[j, j]

    mergemat = pd.DataFrame(data=mergemat, index=sorted(clusters.keys()),
                            columns=sorted(clusters.keys()))
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
    o1 = 0
    o2 = 1
    mm = mergemat.iloc[o1, o2]
    for i in range(0, mergemat.index.size):
        for j in range(i, mergemat.index.size):
            if (j > i) and (mm < mergemat.iloc[i, j]):
                o1 = i
                o2 = j
                mm = mergemat.iloc[i, j]

    o1 = mergemat.index[o1]
    o2 = mergemat.index[o2]
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
    mergemat.loc[k, k] = vit_cluster_liklihood(similarity, clusters[k], model)
    for cid in mergemat.index:
        if cid != k:
            nlp = vit_joint_cluster_liklihood(similarity, clusters[cid],
                                              clusters[k], model)

            mergemat.loc[cid, k] = nlp - mergemat.loc[k, k] - mergemat.loc[cid, cid]

    # assemble column of linkage matrix
    col = np.array([o1, o2, -1 * mm, ksize])
    col = col.reshape(1, 4)
    return col, clusters, mergemat


def vit_linkage(data, model, start_clusters=None):
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

    vpaths = {seq: model.predict(data.loc[seq, :])
              for seq in data.index.values}

    # similarity = viterbi_pair_similarity(data, vpaths, model)
    mergemat = gen_mergemat_vit(vpaths, clusters, model)
    # cluster merging
    k = n  # new cluster id
    while len(clusters) > 1:
        # perform merge
        col, clusters, mergemat = \
            vit_make_merge(clusters, mergemat, k, vpaths, model)
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

genefile = sys.argv[1]
model_dir = sys.argv[2]
agglom_type = 'reg'

gc, mt, track = load_data()
genes = load(open(genefile,  'r'))
data = gc.data.loc[genes, :]
sequences = data.as_matrix()

model_path = '/'.join(model_dir.split('/') + ['model'])
model = HiddenMarkovModel.from_json(model_path)

Z, k = heap_linkage(data.iloc[:100, :], model)
linkage_path = '/'.join(model_dir.split('/') + ['linkage.p'])
dump(Z, open(linkage_path, 'wb'))
# linkage_path = '/'.join(model_dir.split('/') + ['linkage.p'])
# dump(Z, open(linkage_path, 'wb'))
"""
if agglom_type == 'vit':
    Z, k = vit_linkage(data, model)

    linkage_path = '/'.join(model_dir.split('/') + ['vit_linkage.p'])
    dump(Z, open(linkage_path, 'wb'))

else:
    Z, k = heap_linkage(data, model)

    linkage_path = '/'.join(model_dir.split('/') + ['linkage.p'])
    dump(Z, open(linkage_path, 'wb'))
"""

# k = 100
# Z, mm, c, d, s = fast_linkage(data.iloc[:50, :], model)
# linkage_path = '/'.join(model_path.split('/')[:-1] + ['linkage.p'])
# dump(Z, open(linkage_path, 'wb'))
