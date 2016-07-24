import addpath
import numpy as np
import pandas as pd
from pomegranate import HiddenMarkovModel
from src.tools.cluster_evaluation import (dunn_index, davies_bouldin_index)
import hmm.khmm as khmm
from hmm.load_data import load_data
from src.tools.helpers import HeatMap
import glob
import sys
from pickle import dump


def run(cluster_directory_root, depth, plottype):

    # load data
    gc, mt, track = load_data(None, 0)
    data = pd.concat([gc.data, mt.data])

    labels = data.index.values
    pos_labels = labels + '+'
    neg_labels = labels + '-'
    pos_data = pd.DataFrame(data=data.as_matrix(), index=pos_labels,
                            columns=data.columns.values)
    neg_data = pd.DataFrame(data=data.as_matrix(), index=neg_labels,
                            columns=data.columns.values)

    data = pd.concat([data, pos_data, neg_data])

    generic_dir = cluster_directory_root.split('/') + (['*'] * depth)
    generic_dir = ('/').join(generic_dir)
    cluster_directories = \
        glob.glob(generic_dir)

    clusterings = {}
    clusterings_models = {}
    for cluster_dir in cluster_directories:
        try:
            clustering_id = cluster_dir.split('/')[-1:][0]
            # read final clusters
            clusters = {}
            filepath = '/'.join(cluster_dir.split('/') + ['assignments.txt'])
            lines = (open(filepath, 'r').read().splitlines())
            l = 0
            while l < len(lines):
                cluster_name = lines[l]
                cluster_members = lines[l + 1].split('\t')
                clusters[cluster_name] = cluster_members
                l += 4

            clusterings[clustering_id] = clusters

            # load models
            models = {}
            model_files = glob.glob(cluster_dir + '/*')
            for model_file in model_files:
                try:
                    model_id = model_file.split('/')[-1:][0]
                    json = open(model_file).read()
                    models[model_id] = HiddenMarkovModel.from_json(json)
                    print 'model loaded from: ', model_file
                except:
                    pass
            clusterings_models[clustering_id] = models
        except:
            pass

    background = set()
    for clustering in clusterings.itervalues():
        for cid, members in clustering.iteritems():
            background.update(set(members))

    background = list(background)
    # data = data.loc[background, :]

    # generate ranomd clusterings of the same size k as our models
    random_clusterings = {}

    for clustering_id, clustering in clusterings.iteritems():
        source = np.array(background)
        random_assignments = np.random.choice(len(clustering), source.size)
        random_clusters = {}
        for i, cluster_id in enumerate(clustering.iterkeys()):
            random_clusters[cluster_id] = \
                source[np.where(random_assignments == i)[0]].tolist()
        random_clusterings[clustering_id] = random_clusters

    # run dunn and davies_bouldin for clusterings and random permutations
    rand_dunn = report_dunn(random_clusterings, clusterings_models, data)
    savename = cluster_directory_root + 'dunn_index_random'
    dump(rand_dunn, open(savename, 'w'))

    rand_davies = report_davies_bouldin(random_clusterings, clusterings_models,
                                        data)
    savename = cluster_directory_root + 'davies_bouldin_index_random'
    dump(rand_davies, open(savename, 'w'))

    if plottype == 'none':
        pass

    elif plottype == 'kn_grid':

        rand_dunn_df = pd.DataFrame()
        rand_davies_df = pd.DataFrame()

        for clustering_id, clustering in clusterings.iteritems():
            cid = clustering_id.replace('k', '_'). \
                replace('n', '_').split('_')
            m = cid[0]
            k = int(cid[1])
            n = int(cid[2])

            rand_dunn_df.loc[k, n] = rand_dunn[clustering_id]
            rand_davies_df.loc[k, n] = rand_davies[clustering_id]

        rand_davies_df = rand_davies_df.fillna(0)
        rand_dunn_df = rand_dunn_df.fillna(0)

        rand_dunn_df = rand_dunn_df.sort_index().sort_index(1)
        rand_davies_df = rand_davies_df.sort_index().sort_index(1)

        odir = cluster_directory_root
        title = 'RANDOM_' + str(m) + ': Dunn Index'
        HeatMap(rand_dunn_df.as_matrix(), rand_dunn_df.index.values,
                rand_dunn_df.columns.values,
                title=title, odir=odir)

        odir = cluster_directory_root
        title = 'RANDOM_' + str(m) + ': Davies-Bouldin Index'
        HeatMap(rand_davies_df.as_matrix(), rand_davies_df.index.values,
                rand_davies_df.columns.values,
                title=title, odir=odir)

    elif plottype == 'row':
        rand_dunn_df = pd.Series()
        rand_davies_df = pd.Series()

        for clustering_id, clustering in clusterings.iteritems():
            rand_dunn_df.loc[clustering_id] = rand_dunn[clustering_id]
            rand_davies_df.loc[clustering_id] = rand_davies[clustering_id]

        rand_davies_df = rand_davies_df.fillna(0)
        rand_dunn_df = rand_dunn_df.fillna(0)

        rand_dunn_df = rand_dunn_df.sort_index()
        rand_davies_df = rand_davies_df.sort_index()

        odir = cluster_directory_root
        title = 'RANDOM' + ': Dunn Index'
        HeatMap(rand_dunn_df.as_matrix().reshape(-1, 1),
                rand_dunn_df.index.values,
                [' '], title=title, odir=odir, cmin=0, cmax=.5)

        odir = cluster_directory_root
        title = 'RANDOM' + ': Davies-Bouldin Index'
        HeatMap(rand_davies_df.as_matrix().reshape(-1, 1),
                rand_davies_df.index.values,
                [' '], title=title, odir=odir, cmin=5, cmax=10)

    return clusterings, clusterings_models


def report_dunn(clusterings, clusterings_models, data):
    dunn = {}
    for clustering_id, clustering in clusterings.iteritems():
        # get models
        models = clusterings_models[clustering_id]

        # distance calculation settings
        """
        intra_func = khmm.sampled_intra_distance
        inter_func = khmm.inter_distance
        intra_args = {'distance_func': khmm.viterbi_distance,
                      'models': models, 'data': data, 'stat': 'mean'}
        inter_args = {'models': models, 'sample_length': 43, 'num_samples': 43}
        """

        intra_func = khmm.intra_distance
        inter_func = khmm.averaged_inter_distance
        intra_args = {'distance_func': khmm.viterbi_distance,
                      'models': models, 'data': data, 'stat': 'mean'}
        inter_args = {'models': models, 'clusters': clustering,
                      'distance_func': intra_func, 'distance_args': intra_args}

        print 'Calculating Dunn Index for ', clustering_id
        di = dunn_index(clustering, intra_func, intra_args, inter_func,
                        inter_args)
        dunn[clustering_id] = di
        print clustering_id, 'dunn index = ', di

    return dunn


def report_davies_bouldin(clusterings, clusterings_models, data):
    davies_bouldin = {}
    for clustering_id, clustering in clusterings.iteritems():
        # get models
        models = clusterings_models[clustering_id]
        # distance calculation settings
        """
        intra_func = khmm.sampled_intra_distance
        inter_func = khmm.inter_distance
        intra_args = {'distance_func': khmm.viterbi_distance,
                      'models': models, 'data': data, 'stat': 'mean'}
        inter_args = {'models': models, 'sample_length': 43, 'num_samples': 43}
        """
        intra_func = khmm.intra_distance
        inter_func = khmm.averaged_inter_distance
        intra_args = {'distance_func': khmm.viterbi_distance,
                      'models': models, 'data': data, 'stat': 'mean'}
        inter_args = {'models': models, 'clusters': clustering,
                      'distance_func': intra_func, 'distance_args': intra_args}

        print 'Calculating Davies-Bouldin Index for ', clustering_id
        db = davies_bouldin_index(clustering, intra_func, intra_args,
                                  inter_func, inter_args)
        davies_bouldin[clustering_id] = db
        print clustering_id, 'davies-bouldin index = ', db
    return davies_bouldin

if __name__ == "__main__":
    directory = sys.argv[1]
    depth = int(sys.argv[2])
    plottype = sys.argv[3]
    print 'genes'
    print directory
    run(directory, depth, plottype)
