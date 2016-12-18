import addpath
import glob
from hmm.load_data import load_data
import numpy as np
import pandas as pd
import sys
import itertools
import addpath
from pickle import load, dump
from pandas.tools.plotting import autocorrelation_plot
import os
from matplotlib import pyplot as plt
import seaborn as sns
import math

def tragectory_plot(clusters, data, savedir=None):
    n = len(clusters)
    x = math.sqrt(n)
    x = int(x)

    y = 1
    while x * y < n:
        y += 1

    pallete = sns.color_palette("hls", n)
    k = 0

    timepoints = np.arange(data.shape[1])
    plt.figure()
    for cid, cluster in clusters.iteritems():
        mean = data.loc[cluster, :].mean()
        std = data.loc[cluster, :].std()
        plt.plot(timepoints, mean, color=pallete[k])
        # plt.fill_between(timepoints, mean - 2 * std, mean + 2 * std,
        #                 color=pallete[k], alpha=0.1)
        plt.fill_between(timepoints, mean - std, mean + std,
                         color=pallete[k], alpha=0.3)
        # sns.tsplot(data=data.loc[cluster, :].as_matrix(), err_style='ci_band', color=pallete[k], ci=[68, 95])
        k += 1
    savepath = '/'.join(directory.split('/') + ['tragectories.png'])
    plt.savefig(savepath)
    plt.close()

    fig, axarr = plt.subplots(x, y, sharex=True, sharey=True, figsize=(2*x, 2*y))
    i = 0
    j = 0
    k = 0
    for cid, cluster in clusters.iteritems():
        ax = axarr[i, j]
        mean = data.loc[cluster, :].mean()
        std = data.loc[cluster, :].std()
        ax.plot(timepoints, mean, color=pallete[k])
        ax.fill_between(timepoints, mean - 2 * std, mean + 2 * std,
                        color=pallete[k], alpha=0.1)
        ax.fill_between(timepoints, mean - std, mean + std,
                        color=pallete[k], alpha=0.3)
        # sns.tsplot(data=data.loc[cluster, :].as_matrix(), err_style='ci_band', color=pallete[k], ax=axarr[i, j], ci=[68, 95])
        k += 1
        i = (i + 1) % x
        if i == 0:
            j = (j + 1) % y

    savepath = '/'.join(directory.split('/') + ['tragectory_grid.png'])
    plt.savefig(savepath)
    plt.close()

def autocorr_plot(clusters, data, savedir=None):
    n = len(clusters)
    x = math.sqrt(n)
    x = int(x)

    y = 1
    while x * y < n:
        y += 1

    pallete = sns.color_palette("hls", n)
    k = 0

    plt.figure()
    autocorrelation_plot(data.T)
    savepath = '/'.join(directory.split('/') + ['all_autocorr.png'])
    plt.savefig(savepath)
    plt.close()

    fig, axarr = plt.subplots(x, y, sharex=True, sharey=True)
    i = 0
    j = 0
    k = 0
    for cid, cluster in clusters.iteritems():
        ax = axarr[i, j]
        autocorrelation_plot(data.loc[cluster, :].T, ax=ax)
        k += 1
        i = (i + 1) % x
        if i == 0:
            j = (j + 1) % y

    savepath = '/'.join(directory.split('/') + ['incluster_autocorr.png'])
    plt.savefig(savepath)
    plt.close()

sns.set_style("whitegrid", {'axes.grid': False})
directory = sys.argv[1]
data_file = sys.argv[2]
genefile = sys.argv[3]
assignment_file = sys.argv[4]

data = pd.DataFrame.from_csv(data_file, sep=' ')
genes = load(open(genefile,  'r'))
data = data.loc[genes, :]
data = ((data.T - data.T.mean()) / data.T.std()).T

assignment_path = '/'.join(directory.split('/') + [assignment_file])
clusters = load(open(assignment_path, 'r'))
tragectory_plot(clusters, data, directory)
autocorr_plot(clusters, data, directory)