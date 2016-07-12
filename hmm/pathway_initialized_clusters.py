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

metab_id_file = '../data/my_connectome/metab_ids.txt'
metab_id_table = np.genfromtxt(metab_id_file, delimiter='\t', skip_header=1, dtype=str)

metab = metab.iloc[np.where(metab_id_table[:,2] != 'Not Found')[0], :]
kegg_ids = metab_id_table[np.where(metab_id_table[:,2] != 'Not Found')[0], 2]

mt = Preprocessing(metab.as_matrix(), kegg_ids, metab.columns.values,
                   transpose=False)
mt.log_transform(0)
mt.scale()
mt.clean(components=[0], update_data=True)

kegg_genesets = (open('../src/gsea/KEGG_genes', 'r').read().splitlines())
kegg_metabsets = (open('../src/gsea/KEGG_metabolites', 'r').read().splitlines())


gene_pathway_members = []
gene_pathway_names = []
for i in range(len(kegg_genesets)):
    gene_pathway_names.append(kegg_genesets[i].split('\t')[0])
    pathway_genes = kegg_genesets[i].split('\t')[2:]
    p_genes = []
    for gene in pathway_genes:
        if gene in gc.data.index:
            p_genes.append(gene)
    gene_pathway_members.append(p_genes)


metab_pathway_members = []
metab_pathway_names = []
for i in range(len(kegg_metabsets)):
    metab_pathway_names.append(kegg_metabsets[i].split('\t')[0])
    pathway_metabs = kegg_metabsets[i].split('\t')[2:]
    p_metabs = []
    for metab in pathway_metabs:
        if metab in mt.data.index:
            p_metabs.append(metab)
    metab_pathway_members.append(p_metabs)

# for now, remove genes and metabolites shared by pathways
# these will get assigned to a cluster later
unique_gene_pathway_members = []
for pathway in gene_pathway_members:
    unique = set(pathway).difference(*[set(otherpathway) for otherpathway in
                                     gene_pathway_members if
                                     otherpathway != pathway])
    unique_gene_pathway_members.append(list(unique))

unique_metab_pathway_members = []
for pathway in metab_pathway_members:
    unique = set(pathway).difference(*[set(otherpathway) for otherpathway in
                                     metab_pathway_members if
                                     otherpathway != pathway])
    unique_metab_pathway_members.append(list(unique))

# we'll make a cluster for each pathway that has combined at least z unique
# genes and metabolites
z = 1
pathways = list(set(gene_pathway_names).union(set(metab_pathway_names)))
num_unique = np.zeros(len(pathways))
for i, pathway in enumerate(pathways):
    try:
        g_index = gene_pathway_names.index(pathway)
        g = len(unique_gene_pathway_members[g_index])
    except:
        g = 0

    try:
        m_index = metab_pathway_names.index(pathway)
        m = len(unique_metab_pathway_members[m_index])
    except:
        m = 0

    num_unique[i] = g + m

pathways = np.array(pathways)
pathways = pathways[num_unique > z]
pathways = pathways.tolist()

k = len(pathways) # number of models
# n = 5 # number of states
m = gc.data.index.size
models = np.empty(0)
for i in range(k):
    # models = np.append(models, hmm.GaussianHMM(n_components = n))
    models = np.append(models, hard_leftright(43, 1))

noise = hmm.GaussianHMM(n_components=1)

msequences, mlengths, mlabels = df_to_sequence_list(mt.data)
gsequences, glengths, glabels = df_to_sequence_list(gc.data.iloc[:m, :])

sequences = np.concatenate((msequences, gsequences))
lengths = np.concatenate((mlengths, glengths))

# metabolites are either not assigned to a cluster (-1)
# or they are initialized to one of the pathway clusters
# if that metabolite is unique to that pathway
massignments = np.array([k+1] * mlengths.size)
for i, metab in enumerate(mlabels):
    for j, pathway in enumerate(pathways):
        if pathway in metab_pathway_names:
            index = metab_pathway_names.index(pathway)
            if metab in unique_metab_pathway_members[index]:
                massignments[i] = j

# genes are initialized to a pathway cluster if that gene is unique
# to that pathway, otherwise they are assigned to the noise cluster
# which is indexed at k
gassignments = np.array([k] * glengths.size)
for i, gene in enumerate(glabels):
    for j, pathway in enumerate(pathways):
        if pathway in gene_pathway_names:
            index = gene_pathway_names.index(pathway)
            if gene in unique_gene_pathway_members[index]:
                gassignments[i] = j

assignments = np.concatenate((massignments, gassignments))

# set as initialization, no genes or metabolites are fixed
mfixed = np.array([0] * mlengths.size)
gfixed = np.array([0] * glengths.size)
fixed = np.concatenate((mfixed, gfixed))

labels = np.concatenate((mlabels, glabels))

eps = 1e-5
max_iter = 500

noise.fit(gsequences, glengths) # fit noise model to gene expression
# noise._covars_ = noise._covars_ / 4

print np.bincount(assignments)
models, assignments, converged = cluster(models, np.array([noise]), sequences, lengths, assignments, fixed, eps, max_iter, save_name='kegg_init_convergence.txt')
print np.bincount(assignments)

dump([labels, assignments, lengths, fixed, models, noise, eps, converged],
     open('pathway_init_cluster.p'))

"""
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
"""