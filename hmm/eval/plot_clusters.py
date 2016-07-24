import addpath
import pandas as pd
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
from hmm.load_data import load_data
import os
from pomegranate import HiddenMarkovModel


def gen_cluster_plots(cluster_directory_root, depth):
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
    for clustering_id, clustering in clusterings.iteritems():
        for model_id, members in clustering.iteritems():
            sequences = data.loc[members, :]
            pltdir = '/'.join(cluster_directory_root.split('/') + ['plots'])

            # make line plots directory
            if not os.path.isdir(pltdir + '/line'):
                print "Creating directory...", pltdir
                os.mkdir(pltdir + '/line')

            savename = pltdir + '/line/' + model_id + '_lineplot'

            plt_title = model_id + ' Line Plot'
            ax = sequences.T.plot(legend=False, rot=2)
            ax.set_title(plt_title)
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Normalized Expression')

            print 'Saving: ', savename
            fig = ax.get_figure()
            fig.savefig(savename)
            fig.clear()

            # make autocorr plots directory
            if not os.path.isdir(pltdir + '/autocorr'):
                print "Creating directory...", pltdir
                os.mkdir(pltdir + '/autocorr')

            savename = pltdir + '/autocorr/' + model_id + '_autocorr'

            plt_title = model_id + ' Autocorr Plot'
            for seq in sequences.index:
                ax = autocorrelation_plot(sequences.loc[seq])
            ax.set_title(plt_title)

            print 'Saving: ', savename
            fig = ax.get_figure()
            fig.savefig(savename)
            fig.clear()

            # make lag plots directory
            if not os.path.isdir(pltdir + '/lag'):
                print "Creating directory...", pltdir
                os.mkdir(pltdir + '/lag')

            from pylab import *
            NUM_COLORS = len(members)
            cm = get_cmap('gist_rainbow')
            colors = []
            for i in range(NUM_COLORS):
                colors.append(cm(1.*i/NUM_COLORS))

            savename = pltdir + '/lag/' + model_id + '_lagplot'

            plt_title = model_id + ' Lag Plot'
            for i, seq in enumerate(sequences.index):
                ax = lag_plot(sequences.loc[seq], c=colors[i])
            ax.set_title(plt_title)

            print 'Saving: ', savename
            fig = ax.get_figure()
            fig.savefig(savename)
            fig.clear()

            """
            lag_title = model_id + ' Lag Plot'
            plt1 = sequences.plot()
            plt1.set_title()
            """
if __name__ == "__main__":
    directory = sys.argv[1]
    depth = int(sys.argv[2])
    print directory
    gen_cluster_plots(directory, depth)
