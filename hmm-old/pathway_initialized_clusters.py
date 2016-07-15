import addpath
from src.khmm import hard_leftright, soft_leftright, soft_leftright_forced, df_to_sequence_list, cluster
import numpy as np
import pandas as pd
from pickle import dump
from hmmlearn import hmm
from load_data import gc, mt
from load_kegg_pathways import (pathways, metab_pathway_names,
                                gene_pathway_names,
                                unique_metab_pathway_members,
                                unique_gene_pathway_members)

# m = gc.data.index.size  # restricts number of genes, used for local testing
m = 1000

k = len(pathways)  # number of models
# n = 5 # number of states

models = np.empty(0)
for i in range(k):
    # models = np.append(models, hmm.GaussianHMM(n_components = n))
    models = np.append(models, soft_leftright_forced(10, 1))

noise = hmm.GaussianHMM(n_components=1)

mendstate = pd.DataFrame({'es': [1e3] * mt.data.index.size},
                        index=mt.data.index)
msequences, mlengths, mlabels = df_to_sequence_list(mt.data.join(mendstate))


gendstate = pd.DataFrame({'es': [1e3] * m},
                        index=gc.data.iloc[:m, :].index)
gsequences, glengths, glabels = df_to_sequence_list(gc.data.iloc[:m, :].join(gendstate))

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

noise.fit(gsequences, glengths)  # fit noise model to gene expression
# noise._covars_ = noise._covars_ / 2

print np.bincount(assignments)
models, assignments, converged = cluster(models, np.array([noise]), sequences,
                                         lengths, assignments, fixed, eps,
                                         max_iter,
                                         save_name='kegg_init_convergence.txt')
print np.bincount(assignments)

dump([labels, assignments, lengths, fixed, models, noise, eps, converged],
     open('pathway_init_cluster.p', 'wb'))
