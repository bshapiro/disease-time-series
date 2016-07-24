import addpath
import numpy as np
import pandas as pd
from pomegranate import HiddenMarkovModel
from src.tools.cluster_evaluation import (dunn_index, davies_bouldin_index)
import hmm.khmm as khmm
from hmm.load_data import load_data
import glob
import sys
from pickle import dump


def prep(cluster_directory_root, depth, plottype='row'):

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
        random_assignments = np.random.choice(len(clustering), source.size)
        random_assignments = np.random.choice(len(clustering), source.size)
        random_assignments = np.random.choice(len(clustering), source.size)
        random_clusters = {}
        for i, cluster_id in enumerate(clustering.iterkeys()):
            random_clusters[cluster_id] = \
                source[np.where(random_assignments == i)[0]].tolist()
        random_clusterings[clustering_id] = random_clusters

    return clusterings, random_clusterings, clusterings_models, data


def dunn_davies_plots(cluster_directory_root, clusterings, clusterings_models,
                      data, prefix, plottype):
        # run dunn and davies_bouldin for clusterings and random permutations
        logprob = report_total_logprob(clusterings, clusterings_models,
                                         data)
        savename = cluster_directory_root + prefix + '_log_probs'
        dump(logprob, open(savename, 'w'))

        dunn = report_dunn(clusterings, clusterings_models, data)
        savename = cluster_directory_root + prefix + '_dunn_index'
        dump(dunn, open(savename, 'w'))

        davies = report_davies_bouldin(clusterings, clusterings_models, data)
        savename = cluster_directory_root + prefix + '_davies_bouldin_index'
        dump(davies, open(savename, 'w'))

        if plottype == 'none':
            pass

        """
        elif plottype == 'kn_grid':
            dunn_df = pd.DataFrame()
            davies_df = pd.DataFrame()
            logprob_df = pd.DataFrame()

            for clustering_id, clustering in clusterings.iteritems():
                cid = clustering_id.replace('k', '_'). \
                    replace('n', '_').split('_')
                k = int(cid[1])
                n = int(cid[2])

                dunn_df.loc[k, n] = dunn[clustering_id]
                davies_df.loc[k, n] = davies[clustering_id]
                logprob_df.loc[k, n] = logprob[clustering_id]

            davies_df = davies_df.fillna(0)
            dunn_df = dunn_df.fillna(0)
            logprob_df = logprob_df.fillna(0)

            dunn_df = dunn_df.sort_index().sort_index(1)
            davies_df = davies_df.sort_index().sort_index(1)
            logprob_df = logprob_df.sort_index().sort_index(1)

            odir = cluster_directory_root
            title = prefix + ': Dunn Index'
            HeatMap(dunn_df.as_matrix(), dunn_df.index.values,
                    dunn_df.columns.values,
                    title=title, odir=odir)

            odir = cluster_directory_root
            title = prefix + ': Davies-Bouldin Index'
            HeatMap(davies_df.as_matrix(), davies_df.index.values,
                    davies_df.columns.values,
                    title=title, odir=odir)

            odir = cluster_directory_root
            title = prefix + ': Average Log Prob'
            HeatMap(logprob_df.as_matrix(), logprob_df.index.values,
                    logprob_df.columns.values,
                    title=title, odir=odir)

        if plottype == 'row':
            dunn_df = pd.Series()
            davies_df = pd.Series()
            logprob_df = pd.Series()

            for clustering_id, clustering in clusterings.iteritems():
                dunn_df.loc[clustering_id] = dunn[clustering_id]
                davies_df.loc[clustering_id] = davies[clustering_id]
                logprob_df.loc[clustering_id] = logprob[clustering_id]

            davies_df = davies_df.fillna(0)
            dunn_df = dunn_df.fillna(0)
            logprob_df = logprob_df.fillna(0)

            dunn_df = dunn_df.sort_index()
            davies_df = davies_df.sort_index()
            logprob_df = logprob_df.sort_index()

            odir = cluster_directory_root
            title = prefix + ': Dunn Index'
            HeatMap(dunn_df.as_matrix().reshape(-1, 1), dunn_df.index.values,
                    [' '], title=title, odir=odir, cmin=0, cmax=1)

            odir = cluster_directory_root
            title = prefix + ': Davies-Bouldin Index'
            HeatMap(davies_df.as_matrix().reshape(-1, 1),
                    davies_df.index.values,
                    [' '], title=title, odir=odir, cmin=0, cmax=5)

            odir = cluster_directory_root
            title = prefix + ': Average Log Prob'
            HeatMap(logprob_df.as_matrix().reshape(-1, 1),
                    logprob_df.index.values,
                    [' '], title=title, odir=odir)"""


def report_dunn(clusterings, clusterings_models, data):
    dunn = {}
    for clustering_id, oclustering in clusterings.iteritems():
        # get models
        clustering = oclustering.copy()
        clustering.pop('noise')
        models = clusterings_models[clustering_id].copy()
        models.pop('noise')

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
    for clustering_id, oclustering in clusterings.iteritems():
        # get models
        clustering = oclustering.copy()
        clustering.pop('noise')
        models = clusterings_models[clustering_id].copy()
        models.pop('noise')
        # distance calculation settings
        """
        intra_func = khmm.sampled_intra_distance
        inter_func = khmm.inter_distance
        intra_args = {'distance_func': khmm.viterbi_distance,
                      'models': models, 'data': data, 'stat': 'mean'}
        inter_args = {'models': models, 'sample_length': 43, 'num_samples': 43}
        """
        """
        intra_func = khmm.intra_distance
        inter_func = khmm.averaged_inter_distance
        intra_args = {'distance_func': khmm.weighted_distance,
                      'models': models, 'data': data, 'stat': 'mean'}
        inter_args = {'models': models, 'clusters': clustering,
                      'distance_func': intra_func, 'distance_args': intra_args}
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


def report_total_logprob(clusterings, clusterings_models, data):
    """
    Liklihood of the entire model collection normalized by the number
    of succesfull calculations. In case of random assignment sometimes
    models get assigned to sequences that are incompatible
    """
    logprobs = {}
    noise_model = None
    for clustering_id, clustering in clusterings.iteritems():
        models = clusterings_models[clustering_id]
        total = 0
        count = 0
        missed = 0
        for model_id, members in clustering.iteritems():
            if model_id == 'noise' and noise_model is None:
                noise_model = models[model_id]
            model = models[model_id]
            for member in members:
                try:
                    lp = model.log_probability(data.loc[member, :].as_matrix())
                    total += lp
                    count += 1
                except:
                    print 'Sequence - Model mismatch, ignoring'
                    missed += 1

        print missed, ' missed out of: ', (missed + count)
        try:
            alp = total
        except:
            alp = 0
        logprobs[clustering_id] = alp

    logprobs['NOISE'] = noise_model.summarize(data.as_matrix())

    return logprobs

if __name__ == "__main__":
    directory = sys.argv[1]
    depth = int(sys.argv[2])
    cluster_version = sys.argv[3]
    plottype = sys.argv[4]
    print 'genes'
    print directory
    clusterings, random_clusterings, clusterings_models, data = \
        prep(directory, depth, plottype)
    if cluster_version == 'null':
        c = random_clusterings
        prefix = 'RANDOM'
    else:
        c = clusterings
        prefix = 'TRUE'
    dunn_davies_plots(directory, c, clusterings_models, data, prefix, plottype)
