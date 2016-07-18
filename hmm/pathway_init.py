import addpath
import numpy as np
from pomegranate import NormalDistribution, HiddenMarkovModel
from khmm import df_to_sequence_list, cluster, init_gaussian_hmm
from collections import defaultdict
from load_data import load_data
from load_kegg_pathways import load_kegg_pathways

m = 1000  # restricts number of genes, used for local testing
gc, mt, track = load_data(m)
state_range = [5, 10, 25, 50, 100]
z_range = [3, 5, 10, 20]

msequences, mlabels = df_to_sequence_list(mt.data)
gsequences, glabels = df_to_sequence_list(gc.data.iloc[:m, :])

sequences = np.concatenate((msequences, gsequences))
labels = np.concatenate((mlabels, glabels))

# noise model trained on all data once
# genes/metabolites will be assigned to noise model if other models
# fail to model it better
noise_dist = [NormalDistribution(0, 1)]
noise_trans = np.array([[1]])
starts = np.array([1])
noise = HiddenMarkovModel.from_matrix(noise_trans, noise_dist, starts)
noise.freeze_distributions()


odir_base = 'pathway_init_fixed'  # directory to save files to
max_iter = 500  # max iterations
eps = 1e-6  # convergence threshold

for n in state_range:
    for z in z_range:
        pathways, metab_pathway_names, gene_pathway_names, \
         unique_metab_pathway_members, unique_gene_pathway_members = \
         load_kegg_pathways(gc, mt, z)

        k = len(pathways)

        # number of pathways = number of models
        # metabolites/genes are assigned to a cluster pathway if they are
        # unique to that pathway. otherwise assign to noise
        assignments = defaultdict(list)
        """
        for pathway in pathways:
            mindex = metab_pathway_names.index(pathway)
            gindex = metab_pathway_names.index(pathway)
            members = []
            for mset in mindex:
                members.append(mset)
            for gset in gindex:
                members.append(gset)
            label_index = []
            for member in members:
                label_index = np.where(labels == member)[0]
            assignments[pathway].append(label_index)
        """
        for i, sequence in enumerate(labels):
            placed = False
            for j, pathway in enumerate(pathways):
                if pathway in metab_pathway_names:
                    index = metab_pathway_names.index(pathway)
                    if sequence in unique_metab_pathway_members[index]:
                        assignments[pathway].append(i)
                        placed = True
                if pathway in gene_pathway_names:
                    index = gene_pathway_names.index(pathway)
                    if sequence in unique_gene_pathway_members[index]:
                        assignments[pathway].append(i)
                        placed = True
            if not placed:
                assignments['noise'].append(i)

        fixed = np.zeros(labels.size)
        for pathway, members in assignments.iteritems():
            if pathway is not 'noise':
                fixed[members] = 1  # fix known unique pathway members

        # recall z is the threshold set in load_kegg_pathways for minimum # of
        # unique examples in the pathway
        collection_id = 'k-' + str(k) + '_n-' + str(n) + '_z-' + str(z) +  \
                        '_kegg_fixed'
        odir = odir_base + '/' + collection_id

        print 'Learning: ', collection_id
        models = defaultdict(list)
        for pathway, seq_indices in assignments.iteritems():
            model = init_gaussian_hmm(sequences[seq_indices, :], n,
                                      model_id=pathway)
            models[pathway] = model

        models['noise'] = noise

        models, assignments, c = cluster(models=models,
                                         sequences=sequences,
                                         assignments=assignments,
                                         labels=labels, fixed=fixed,
                                         eps=eps, max_it=max_iter,
                                         odir=odir)

        """
        try:
            # initialize models
            models = defaultdict(list)
            for pathway in assignments:
                if pathway is 'noise':
                    models['noise'] = noise
                else:
                    in_model = assignments[pathway]
                    model = init_gaussian_hmm(sequences[in_model, :], n,
                                              model_id=pathway)
                    models[pathway] = model

            models, assignments, c = cluster(models=models,
                                             noise_models=np.array([noise]),
                                             sequences=sequences,
                                             assignments=assignments,
                                             labels=labels, fixed=fixed,
                                             eps=eps, max_it=max_iter,
                                             odir=odir)
        except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            f.close()
        """

odir_base = 'pathway_init_fixed'  # directory to save files to
for n in state_range:
    for z in z_range:
        pathways, metab_pathway_names, gene_pathway_names, \
         unique_metab_pathway_members, unique_gene_pathway_members = \
         load_kegg_pathways(gc, mt, z)

        k = len(pathways)
        # number of pathways = number of models
        # metabolites are either not assigned to a cluster (-1)
        # or they are initialized to one of the pathway clusters
        # if that metabolite is unique to that pathway
        massignments = np.array([k] * mlabels.size)
        for i, metab in enumerate(mlabels):
            for j, pathway in enumerate(pathways):
                if pathway in metab_pathway_names:
                    index = metab_pathway_names.index(pathway)
                    if metab in unique_metab_pathway_members[index]:
                        massignments[i] = j

        # genes are initialized to a pathway cluster if that gene is unique
        # to that pathway, otherwise they are assigned to the noise cluster
        # which is indexed at k
        gassignments = np.array([k] * glabels.size)
        for i, gene in enumerate(glabels):
            for j, pathway in enumerate(pathways):
                if pathway in gene_pathway_names:
                    index = gene_pathway_names.index(pathway)
                    if gene in unique_gene_pathway_members[index]:
                        gassignments[i] = j

        assignments = np.concatenate((massignments, gassignments))
        fixed = np.zeros(assignments.size)

        # recall z is the threshold set in load_kegg_pathways for minimum # of
        # unique examples in the pathway
        collection_id = 'k-' + str(k) + '_n-' + str(n) + '_z-' + str(z) +  \
                        '_kegg_unfixed'
        odir = odir_base + '/' + collection_id

        print 'Learning: ', collection_id

        try:
            # initialize models
            models = np.empty(0)
            for i in range(k):
                in_model = np.where(assignments == i)[0]
                model = init_gaussian_hmm(sequences[np.where(assignments == 0)
                                          [0], :], n, model_id=str(i))
                models = np.append(models, model)

            models, assignments, c = cluster(models=models,
                                             noise_models=np.array([noise]),
                                             sequences=sequences,
                                             assignments=assignments,
                                             labels=labels, fixed=fixed,
                                             eps=eps, max_it=max_iter,
                                             odir=odir)
        except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            f.close()