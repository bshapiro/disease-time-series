import addpath
import glob
from pomegranate import HiddenMarkovModel
from src.tools.cluster_evaluation import (clusterings_conserved_pairs,
                                          clusterwise_conserved_pairs)
import hmm.khmm as khmm
from hmm.load_data import load_data
import numpy as np
import pandas as pd
import sys
import itertools
from pickle import dump

def init(base_dir):
    print base_dir
    cluster_directories = \
        glob.glob(base_dir + '/*')

    initial_clusterings = {}
    clusterings = {}
    clusterings_models = {}
    for cluster_dir in cluster_directories:
        try:
            clustering_id = cluster_dir.split('/')[-1:][0]
            # read initial clusters
            initial_clusters = {}
            filepath = '/'.join(cluster_dir.split('/') + ['init_assignments.txt'])
            lines = (open(filepath, 'r').read().splitlines())
            l = 0
            while l < len(lines):
                cluster_name = lines[l]
                cluster_members = lines[l + 1].split('\t')
                initial_clusters[cluster_name] = cluster_members
                l += 4

            initial_clusterings[clustering_id] = initial_clusters

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
    return initial_clusterings, clusterings


def init_final(initial_clusterings, clusterings, base_dir):
    # initial vs final heatmaps
    for c_id in initial_clusterings.iterkeys():
        c1 = initial_clusterings[c_id]
        c2 = clusterings[c_id]
        odir = base_dir + '/init_final_conservation/'
        conserved = clusterwise_conserved_pairs(c1, c2)
        conserved = conserved.sort_index().sort_index(1)
        filepath = odir + str(c_id) + '_conservation.p'
        dump(conserved, open(filepath, 'w'))

        title = c_id + ': init-final conservation'
        """HeatMap(conserved.as_matrix(), conserved.index.values,
                conserved.columns.values,
                cmin=0, cmax=1, title=title, odir=odir)"""


def final_final(clusterings, base_dir):
    # final vs final heatmaps
    odir = base_dir + '/conservation/'
    pairs = list(itertools.product(clusterings.keys(), clusterings.keys()))
    final_pairs = [pair for pair in pairs]
    for pair in pairs:
        reverse = (pair[1], pair[0])
        if reverse in final_pairs and pair in final_pairs:
            final_pairs.remove(reverse)
    pairs = final_pairs

    print pairs
    for pair in pairs:
        pair = list(pair)
        c_id1 = pair[0]
        c_id2 = pair[1]
        c1 = clusterings[c_id1]
        c2 = clusterings[c_id2]
        conserved = clusterwise_conserved_pairs(c1, c2)
        conserved = conserved.sort_index().sort_index(1)
        filepath = odir + str(c_id1) + '_' + str(c_id2) + '_conservation.p'
        dump(conserved, open(filepath, 'w'))
        """
        title = c_id1 + ', ' + c_id2 + ': conservation'
        HeatMap(conserved.as_matrix(), conserved.index.values,
                conserved.columns.values,
                cmin=0, cmax=1, title=title, odir=odir)"""


def overall(clusterings, base_dir):
    # total cluster heatmaps
    odir = base_dir
    total_conservation = pd.DataFrame()
    for c_id1, c1 in clusterings.iteritems():
        for c_id2, c2 in clusterings.iteritems():
            total_conservation.loc[c_id1, c_id2] = \
                clusterings_conserved_pairs(c1, c2, 'first')

    title = 'total conservation'

    total_conservation = total_conservation.sort_index().sort_index(1)
    filepath = base_dir + '/conservation.p'
    dump(total_conservation, open(filepath, 'w'))
    """HeatMap(total_conservation.as_matrix(),
            total_conservation.index.values,
            total_conservation.columns.values,
            cmin=0, cmax=None, title=title, odir=odir)"""

if __name__ == "__main__":
    base_dir = sys.argv[1]
    initial_clusterings, clusterings = init(base_dir)
    overall(clusterings, base_dir)
    final_final(clusterings, base_dir)
    # init_final(initial_clusterings, clusterings, base_dir)
