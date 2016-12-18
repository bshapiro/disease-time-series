import addpath
import pandas as pd
import numpy as np
from pomegranate import HiddenMarkovModel
from pickle import load, dump
from collections import defaultdict, OrderedDict
from itertools import combinations
from heapq import heappush, heappop, heapify
import multiprocessing
import sys


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
    probmat = np.ones(sequence_probs[sequence_probs.keys()[0]].shape)
    for sequence in cluster:
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


def gen_mergeheap(share, clusters, sequence_probs, cluster_probs, model):
    print "Generating merged liklihoods..."
    mergeheap = []
    for i, c1 in clusters.iteritems():
        if i in share:
            for j, c2 in clusters.iteritems():
                if j > i:
                    cost = joint_cluster_liklihood(sequence_probs,
                                                   c1, c2, model)
                    cost = cost - cluster_probs[i] - cluster_probs[j]
                    cost *= -1
                    mergeheap.append((cost, (i, j)))

    heapify(mergeheap)
    print len(mergeheap)
    return mergeheap


def compute_merge(share, clusters, sequence_probs, cluster_probs,
                  connection, kill_event):
    # share is this compute processes share of clusters to compute merges for
    # clusters is a list of all clusters
    # connection is connection to main merging task
    # merge_probs is the dictionary of merge probabilities
    # if merge_probs is none we are just starting so generate from sequences
    mergeheap = gen_mergeheap(share, clusters, sequence_probs,
                              cluster_probs, model)

    while len(clusters) > 0:
        merge = (-1, -1)
        merge_cost = np.inf
        while merge[0] not in clusters or merge[1] not in clusters:
            if len(mergeheap) > 0:
                # look at next best merge
                merge_cost, merge = heappop(mergeheap)
            else:
                connection.send((False, -1, -1, -1))
                connection.close()
                kill_event.set()
                return

        # send out best valid merge in process
        connection.send((True, merge_cost, merge, len(mergeheap)))

        # recieve selected merge
        best_merge, k, k_prob = connection.recv()
        if best_merge != -1:
            # update clusters and cluster probs
            clusters[k] = clusters.pop(best_merge[0]) +\
                          clusters.pop(best_merge[1])
            cluster_probs[k] = k_prob
            cluster_probs.pop(best_merge[0])
            cluster_probs.pop(best_merge[1])

            # remove clusters from share if merged
            # add new cluster to share if we are removing from share
            if best_merge[0] in share:
                share.append(k)
                share.remove(best_merge[0])
            if best_merge[1] in share:
                share.remove(best_merge[1])

            # add merges with new cluster if in share
            if k in share:
                for i in clusters.keys():
                    if i != k:
                        c1 = clusters[k]
                        c2 = clusters[i]
                        cost = joint_cluster_liklihood(sequence_probs,
                                                       c1, c2, model)
                        cost = cost - cluster_probs[k] - cluster_probs[i]
                        cost *= -1
                        heappush(mergeheap, (cost, (k, i)))
        else:
            break

    connection.send((False, -1, -1, -1))
    connection.close()
    kill_event.set()
    return

def pick_merge(Z, clusters, compute_processes, connections, sequence_probs, kill_events):
    while len(clusters) > 1:
        # identify best merge from compute processes
        best_merge_cost = np.inf
        best_merge = None
        for i in kill_events.keys():
            if kill_events[i].is_set():
                print 'Process', i, 'ended'
                compute_processes.pop(i)
                connections[i].close()
                connections.pop(i)
                kill_events.pop(i)
            else:
                valid, merge_cost, merge, size = connections[i].recv()
                if valid and merge_cost < best_merge_cost:
                    best_merge_cost = merge_cost
                    best_merge = merge

        # perform merges until no valid merges remain
        if best_merge is not None:
            # add new cluster to clusters and cluster_probs
            k = clusters.keys()[-1] + 1
            clusters[k] = clusters.pop(best_merge[0]) + clusters.pop(best_merge[1])
            k_prob = cluster_liklihood(sequence_probs, clusters[k], model)

            # generate column of linkage matrix
            #col = [best_merge[0], best_merge[1], k_prob * -1,
            #       len(clusters[k])]
            col = [best_merge[0], best_merge[1], float(k - len(sequences)),
                   len(clusters[k])]
            Z.append(col)
            print 'Merged clusters:', \
                best_merge[0], best_merge[1], best_merge_cost, k, len(Z)

            # send back to merge computation processes which merge was selected
            for connection in connections.itervalues():
                connection.send((best_merge, k, k_prob))

        # if no valid merges, kill merge calculation processes
        if best_merge is None:
            # send poison pill to processes
            for connection in connections.itervalues():
                connection.send((-1, 0, 0))

            q = clusters.keys()[-1] + 1
            while len(clusters) > 1:
                # remainder of merges fake merge to satisfy linkage matrix reqs
                print 'Fake merge:', q
                c1 = clusters.keys()[0]
                c2 = clusters.keys()[1]
                clusters[q] = clusters.pop(c1) + clusters.pop(c2)
                col = [c1, c2, q - len(sequence_probs), len(clusters[q])]
                Z.append(col)
                q += 1
    return

def linkage(data, model, num_processes=(multiprocessing.cpu_count()-1)):
    # start with singleton clusters
    n = data.shape[0]
    clusters = {i: [data.index[i]] for i in range(0, n)}
    clusters = OrderedDict(sorted(clusters.items(), key=lambda t: t[0]))

    # sequence probs shared by merge compute processes
    sequence_probs = {seq: model.predict_proba(data.loc[seq, :])
                      for seq in data.index.values}

    cluster_probs = {}
    for c in clusters.keys():
        cluster_probs[c] = cluster_liklihood(sequence_probs, clusters[c],
                                             model)

    # singleton-clusters/sequences assigned to each compute process
    shares = {i: [] for i in range(num_processes)}
    for i, key in enumerate(clusters.keys()):
        shares[(i % num_processes)].append(key)

    # connections between merge compute processes and main merge process
    pipes = {i: multiprocessing.Pipe() for i in range(num_processes - 1)}
    connections = {i: pipes[i][0] for i in pipes}
    kill_events = {i: multiprocessing.Event() for i in range(num_processes - 1)}

    # arguments for each merge compute process
    compute_kw = {i: {'share': shares[i], 'clusters': clusters,
                      'sequence_probs': sequence_probs,
                      'cluster_probs': cluster_probs,
                      'connection': pipes[i][1], 'kill_event': kill_events[i]}
                  for i in range(num_processes - 1)}
    compute_processes = {i: multiprocessing.Process(target=compute_merge,
                                                    name=i,
                                                    kwargs=compute_kw[i])
                         for i in range(num_processes - 1)}

    manager = multiprocessing.Manager()
    Z = manager.list()
    merger_kw = {'Z': Z, 'clusters': clusters,
                 'compute_processes': compute_processes,
                 'connections': connections, 'sequence_probs': sequence_probs,
                 'kill_events': kill_events}
    merge_process = multiprocessing.Process(target=pick_merge, name='merger',
                                            kwargs=merger_kw)

    # Start the processes
    for cp in compute_processes.itervalues():
        cp.start()

    merge_process.start()
    merge_process.join()
    Z = np.array(Z)
    print Z
    return Z


genefile = sys.argv[1]
model_dir = sys.argv[2]
data_file = sys.argv[3]
processes = int(sys.argv[4])

data = pd.DataFrame.from_csv(data_file, sep=' ')
genes = load(open(genefile,  'r'))
data = data.loc[genes, :]
data = ((data.T - data.T.mean()) / data.T.std()).T
sequences = data.as_matrix()

model_path = '/'.join(model_dir.split('/') + ['model'])
model = HiddenMarkovModel.from_json(model_path)

Z = linkage(data, model, processes)
linkage_path = '/'.join(model_dir.split('/') + ['linkage.p'])
dump(Z, open(linkage_path, 'wb'))
