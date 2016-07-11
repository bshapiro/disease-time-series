import addpath
from src.khmm import *
from src.preprocessing import *
from src.representation import *
from src.tools.helpers import *
from datetime import datetime
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump
from hmmlearn import hmm, utils

mc_data = '../data/my_connectome/GSE58122_varstab_data_prefiltered_geo.txt'
mc_genes = '../data/my_connectome/mc_geneset.p'
track_data = '../data/my_connectome/tracking_data.txt'
fd = ['../data/my_connectome/rin.p']
ops = ['>']
th = [6]
rin = pickle.load(open(fd[0]))
goodrin = rin[np.where(rin > th[0])[0]].reshape(-1,1)

data, rl, cl = load_file(mc_data, 'tsv', True, True)
gc = Preprocessing(data, rl, cl)
gc.filter((rin, '>', 6, 1))
gc.clean(components=[0,1,2], regress_out=[(goodrin, 1)], update_data=True)

track_data = '../data/my_connectome/tracking_data.txt'
track = pd.read_csv(track_data, sep='\t', na_values='.', header=0, index_col=0)
track = track[track.loc[:,'rna:rin'] > 6]

metab_file = '../data/my_connectome/metabolomics_raw_data.csv'
metab = pd.read_csv(metab_file, sep='\t', index_col=0, header=4,
                    usecols=[i for i in range(6, 53)].append(0),
                    skiprows=[5,6,7])
metab = metab.iloc[:,5:]

sample_to_date = dict(zip(track.index.tolist(), track['date'].as_matrix()))
date_to_sample = dict(zip(track['date'].as_matrix(), track.index.tolist()))

metab_dates = metab.axes[1].values.tolist()
metab_dates = [datetime.strptime(i, "%m/%d/%Y") for i in metab_dates]
metab_dates = [datetime.strftime(i, "%Y-%m-%d") for i in metab_dates]

metab = pd.DataFrame(metab.as_matrix(), index=metab.axes[0], columns=metab_dates)

metab_samples = []
usable_dates = []
for date in metab_dates:
    if date_to_sample.has_key(date):
        usable_dates.append(date)
        metab_samples.append(date_to_sample[date])

metab = metab.loc[:, usable_dates]
metab = pd.DataFrame(metab.as_matrix(), index=metab.axes[0], columns=metab_samples)

metab = metab.iloc[:106, :]

mt = Preprocessing(metab.as_matrix(), metab.index.values, metab.columns.values,
                   transpose=True)
mt.log_transform(0)
mt.scale()
mt.clean(components=[0], update_data=True)

kegglines = (open('../src/gsea/KEGG.gmt', 'r').read().splitlines())

pathway_members = []
for i in range(len(kegglines)):
    pathway_genes = kegglines[0].split('\t')[2:]
    p_genes = []
    for gene in pathway_genes:
        if gene in gc.data.index:
            p_genes.append(gene)
    pathway_members.append(p_genes)

k = 10 # number of models
n = 5 # number of states
m = 100 # number of genes

models = np.empty(0)
for i in range(k):
    models = np.append(models, hmm.GaussianHMM(n_components = n))

noise = hmm.GaussianHMM(n_components=1)

sequences, lengths, labels = df_to_sequence_list(gc.data.iloc[:m, :])
assignments = np.array([i % k for i in range(m)])
assignments[:(m/2)] = k
fixed = np.array([0] * m)
#fixed[:10] = 1
eps = 1e-5
max_iter = 100

noise.fit(sequences*100, lengths)

shape = noise.covars_.shape
#noise._covars_ = np.array([[1000]])
print np.bincount(assignments)
models, assignments, converged = cluster(models, np.array([noise]), sequences, lengths, assignments, fixed, eps, max_iter, save_name='test2.txt')
print np.bincount(assignments)