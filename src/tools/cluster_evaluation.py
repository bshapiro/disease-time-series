import itertools
import numpy as np
import pandas as pd


def clusterings_conserved_pairs(clustering1, clustering2, denom='all'):
    """
    arguments are two clusterings on the same data where each is a dictionary
    of lists of cluster members.
    eg clustering1 = {c0:[c0_members], ..., ck[ck_members]}
    returns the proportion of coclustered pais between the two
    """
    found_pairs1 = set()
    found_pairs2 = set()
    conserved_pairs = set()
    for cluster1 in clustering1.itervalues():
        pairs1 = set(itertools.combinations(set(cluster1), 2))
        pairs1 = set([frozenset(pair) for pair in pairs1])
        found_pairs1.update(pairs1)
    for cluster2 in clustering2.itervalues():
        pairs2 = set(itertools.combinations(set(cluster2), 2))
        pairs2 = set([frozenset(pair) for pair in pairs2])
        found_pairs2.update(pairs2)

    conserved_pairs = found_pairs1.intersection(found_pairs2)
    found_pairs = found_pairs1.union(found_pairs2)

    prop1 = float(len(conserved_pairs)) / len(found_pairs1)
    prop2 = float(len(conserved_pairs)) / len(found_pairs2)
    prop = float(len(conserved_pairs)) / len(found_pairs)

    out = 0
    if denom == 'first':
        out = prop1
    if denom == 'second':
        out = prop2
    if denom == 'all':
        out = prop
    if denom == 'min':
        out = min(prop1, prop2)
    if denom == 'max':
        out = max(prop1, prop2)

    return out


def clusterwise_conserved_pairs(clustering1, clustering2, denom='all'):
    """
    arguments are two clusterings on the same data where each is a dictionary
    of lists of cluster members.
    eg clustering1 = {c0:[c0_members], ..., ck[ck_members]}
    returns a pandas datafram where index[i, j] is the proportion of shared
    pairs between cluster i of clustering 1 and cluster j of clustering 2
    """
    n = len(clustering1)
    m = len(clustering2)
    conservation = pd.DataFrame(data=np.zeros((n, m)),
                                index=clustering1.keys(),
                                columns=clustering2.keys())
    for c1_id, cluster1 in clustering1.iteritems():
        for c2_id, cluster2 in clustering2.iteritems():
            if denom == 'first':
                conservation.loc[c1_id, c2_id] = \
                    conserved_pairs(cluster1, cluster2)[0]
            if denom == 'second':
                conservation.loc[c1_id, c2_id] = \
                    conserved_pairs(cluster1, cluster2)[1]
            if denom == 'all':
                conservation.loc[c1_id, c2_id] = \
                    conserved_pairs(cluster1, cluster2)[2]
            if denom == 'min':
                conservation.loc[c1_id, c2_id] = \
                    min(conserved_pairs(cluster1, cluster2))
            if denom == 'max':
                conservation.loc[c1_id, c2_id] = \
                    max(conserved_pairs(cluster1, cluster2))

    return conservation


def conserved_pairs(cluster1, cluster2):
    """
    arguments are two clusters, each a list of members
    returns the proportion of shared pairs between the two
    """
    pairs1 = set(itertools.combinations(set(cluster1), 2))
    pairs1 = set([frozenset(pair) for pair in pairs1])

    pairs2 = set(itertools.combinations(set(cluster2), 2))
    pairs2 = set([frozenset(pair) for pair in pairs2])

    conserved_pairs = pairs1.intersection(pairs2)
    found_pairs = pairs1.union(pairs2)

    prop1 = 0
    prop2 = 0
    prop = 0
    if len(pairs1) > 0:
        prop1 = float(len(conserved_pairs)) / len(pairs1)
    if len(pairs2) > 0:
        prop2 = float(len(conserved_pairs)) / len(pairs2)
    if len(found_pairs) > 0:
        prop = float(len(conserved_pairs)) / len(found_pairs)

    return prop1, prop2, prop


def dunn_index(clustering, intra_func, intra_args, inter_func, inter_args):
    """
    general implementation of dunn index. given a dictionary defining
    a clustering {cluster:members}, a intra-cluster function for measuring
    in cluster distance, a inter-cluster function for measuring between cluster
    distances, and dictionary of arguments for each function, will produce a
    dunn index score for the clustering.
    The intra cluster function should take a cluster id and list of member
    the inter cluster function should take a pair of clusters as
    (c1_id, c1_members, c2_id, c2_members, **kwargs)
    """
    intra_scores = []
    inter_scores = []
    for c1_id, members1 in clustering.iteritems():
        intra_scores.append(intra_func(c1_id, members1, **intra_args))
        for c2_id, members2 in clustering.iteritems():
            if c1_id != c2_id:
                inter_scores.append(inter_func(c1_id, c2_id, **inter_args))

    dunn_index = min(inter_scores) / max(intra_scores)
    return dunn_index


def davies_bouldin_index(clustering, intra_func, intra_args, inter_func,
                         inter_args):
    n = len(clustering.keys())
    total = 0
    for c1_id, members1 in clustering.iteritems():
        max_found = 0
        for c2_id, members2 in clustering.iteritems():
            if c1_id != c2_id:
                val = (intra_func(c1_id, members1, **intra_args) +
                       intra_func(c2_id, members2, **intra_args)) / \
                       inter_func(c1_id, c2_id, **inter_args)
                max_found = max(max_found, val)
        total += max_found
    davies_bouldin_index = total / n
    return davies_bouldin_index
