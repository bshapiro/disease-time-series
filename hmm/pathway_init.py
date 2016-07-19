import addpath
import sys
import numpy as np
from pomegranate import NormalDistribution, HiddenMarkovModel
from khmm import df_to_sequence_list, cluster, init_gaussian_hmm
from collections import defaultdict
from load_data import load_data
from load_kegg_pathways import load_kegg_pathways


def init():
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
    return gc, mt, sequences, labels, noise, z_range, state_range


def kegg_init_vit():
    gc, mt, sequences, labels, noise, z_range, state_range = init()
    for n in state_range:
        for z in z_range:
            try:
                pathways, metab_pathway_names, gene_pathway_names, \
                 unique_metab_pathway_members, unique_gene_pathway_members = \
                 load_kegg_pathways(gc, mt, z)

                k = len(pathways)
                # directory to save files to
                odir_base = '../results/khmm/viterbi/kegg_init'
                collection_id = 'k-' + str(k) + '_z-' + str(z) + '_n-' \
                    + str(n) + '_kegg_init'
                odir = odir_base + '/' + collection_id

                # number of pathways = number of models
                # metabolites/genes are assigned to a cluster pathway if
                # unique to that pathway. otherwise assign to noise
                assignments = defaultdict(list)
                for i, sequence in enumerate(labels):
                    placed = False
                    for j, pathway in enumerate(pathways):
                        # if metabolite is unique to cluster, assign
                        if pathway in metab_pathway_names:
                            index = metab_pathway_names.index(pathway)
                            if sequence in unique_metab_pathway_members[index]:
                                assignments[pathway].append(i)
                                placed = True
                        # if gene is unique to cluster, assign
                        if pathway in gene_pathway_names:
                            index = gene_pathway_names.index(pathway)
                            if sequence in unique_gene_pathway_members[index]:
                                assignments[pathway].append(i)
                                placed = True
                    # assign to noise otherwise
                    if not placed:
                        assignments['noise'].append(i)

                # create model
                models = defaultdict(list)
                for pathway, seq_indices in assignments.iteritems():
                    model = init_gaussian_hmm(sequences[seq_indices, :], n,
                                              model_id=pathway)
                    models[pathway] = model

                # add noise model
                models['noise'] = noise

                # perform clustering
                print 'Learning: ', collection_id
                models, assignments, c = cluster(models=models,
                                                 sequences=sequences,
                                                 assignments=assignments,
                                                 labels=labels,
                                                 algorithm='viterbi',
                                                 odir=odir)
            except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()


def kegg_init():
    gc, mt, sequences, labels, noise, z_range, state_range = init()
    for n in state_range:
        for z in z_range:
            try:
                pathways, metab_pathway_names, gene_pathway_names, \
                 unique_metab_pathway_members, unique_gene_pathway_members = \
                 load_kegg_pathways(gc, mt, z)

                k = len(pathways)
                # directory to save files to
                odir_base = '../results/khmm/kegg_init'
                collection_id = 'k-' + str(k) + '_z-' + str(z) + '_n-' \
                    + str(n) + '_kegg_init'
                odir = odir_base + '/' + collection_id

                # number of pathways = number of models
                # metabolites/genes are assigned to a cluster pathway if
                # unique to that pathway. otherwise assign to noise
                assignments = defaultdict(list)
                for i, sequence in enumerate(labels):
                    placed = False
                    for j, pathway in enumerate(pathways):
                        # if metabolite is unique to cluster, assign
                        if pathway in metab_pathway_names:
                            index = metab_pathway_names.index(pathway)
                            if sequence in unique_metab_pathway_members[index]:
                                assignments[pathway].append(i)
                                placed = True
                        # if gene is unique to cluster, assign
                        if pathway in gene_pathway_names:
                            index = gene_pathway_names.index(pathway)
                            if sequence in unique_gene_pathway_members[index]:
                                assignments[pathway].append(i)
                                placed = True
                    # assign to noise otherwise
                    if not placed:
                        assignments['noise'].append(i)

                # create model
                models = defaultdict(list)
                for pathway, seq_indices in assignments.iteritems():
                    model = init_gaussian_hmm(sequences[seq_indices, :], n,
                                              model_id=pathway)
                    models[pathway] = model

                # add noise model
                models['noise'] = noise

                # perform clustering
                print 'Learning: ', collection_id
                models, assignments, c = cluster(models=models,
                                                 sequences=sequences,
                                                 assignments=assignments,
                                                 labels=labels,
                                                 algorithm='baum_welch',
                                                 odir=odir)
            except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()


def kegg_init_fixed_vit():
    gc, mt, sequences, labels, noise, z_range, state_range = init()
    for n in state_range:
        for z in z_range:
            #try:
            pathways, metab_pathway_names, gene_pathway_names, \
             unique_metab_pathway_members, unique_gene_pathway_members = \
             load_kegg_pathways(gc, mt, z)

            k = len(pathways)
            # directory to save files to
            odir_base = '../results/khmm/viterbi/kegg_init_fixed'
            collection_id = 'k-' + str(k) + '_z-' + str(z) + '_n-' \
                + str(n) + '_kegg_init'
            odir = odir_base + '/' + collection_id

            # number of pathways = number of models
            # metabolites/genes are assigned to a cluster pathway if
            # unique to that pathway. otherwise assign to noise
            assignments = defaultdict(list)
            for i, sequence in enumerate(labels):
                placed = False
                for j, pathway in enumerate(pathways):
                    # if metabolite is unique to cluster, assign
                    if pathway in metab_pathway_names:
                        index = metab_pathway_names.index(pathway)
                        if sequence in unique_metab_pathway_members[index]:
                            assignments[pathway].append(i)
                            placed = True
                    # if gene is unique to cluster, assign
                    if pathway in gene_pathway_names:
                        index = gene_pathway_names.index(pathway)
                        if sequence in unique_gene_pathway_members[index]:
                            assignments[pathway].append(i)
                            placed = True
                # assign to noise otherwise
                if not placed:
                    assignments['noise'].append(i)

            # create models
            models = defaultdict(list)
            for pathway, seq_indices in assignments.iteritems():
                model = init_gaussian_hmm(sequences[seq_indices, :], n,
                                          model_id=pathway)
                models[pathway] = model

            # add noise model
            models['noise'] = noise

            # no ties, each sequence is tied only to itself
            tied = {}
            for i, label in enumerate(labels):
                tied[label] = [i]

            # fix non-noise model members to their respective models
            fixed = {}
            for model_id, model in models.iteritems():
                fixed[model_id] = []
                for group_id, group_members in tied.iteritems():
                    if len(set(group_members).intersection(
                               set(assignments[model_id]))) > 0:
                        fixed[model_id].append(group_id)

            fixed['noise'] = []

            # perform clustering
            print 'Learning: ', collection_id
            models, assignments, c = cluster(models=models,
                                             sequences=sequences,
                                             assignments=assignments,
                                             labels=labels,
                                             fixed=fixed, tied=tied,
                                             algorithm='viterbi',
                                             odir=odir)
            """except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()"""

def kegg_init_fixed():
    gc, mt, sequences, labels, noise, z_range, state_range = init()
    for n in state_range:
        for z in z_range:
            try:
                pathways, metab_pathway_names, gene_pathway_names, \
                 unique_metab_pathway_members, unique_gene_pathway_members = \
                 load_kegg_pathways(gc, mt, z)

                k = len(pathways)
                # directory to save files to
                odir_base = '../results/khmm/kegg_init_fixed_vit'
                collection_id = 'k-' + str(k) + '_z-' + str(z) + '_n-' \
                    + str(n) + '_kegg_init'
                odir = odir_base + '/' + collection_id

                # number of pathways = number of models
                # metabolites/genes are assigned to a cluster pathway if
                # unique to that pathway. otherwise assign to noise
                assignments = defaultdict(list)
                for i, sequence in enumerate(labels):
                    placed = False
                    for j, pathway in enumerate(pathways):
                        # if metabolite is unique to cluster, assign
                        if pathway in metab_pathway_names:
                            index = metab_pathway_names.index(pathway)
                            if sequence in unique_metab_pathway_members[index]:
                                assignments[pathway].append(i)
                                placed = True
                        # if gene is unique to cluster, assign
                        if pathway in gene_pathway_names:
                            index = gene_pathway_names.index(pathway)
                            if sequence in unique_gene_pathway_members[index]:
                                assignments[pathway].append(i)
                                placed = True
                    # assign to noise otherwise
                    if not placed:
                        assignments['noise'].append(i)

                # create models
                models = defaultdict(list)
                for pathway, seq_indices in assignments.iteritems():
                    model = init_gaussian_hmm(sequences[seq_indices, :], n,
                                              model_id=pathway)
                    models[pathway] = model

                # add noise model
                models['noise'] = noise

                # no ties, each sequence is tied only to itself
                tied = {}
                for i, label in enumerate(labels):
                    tied[label] = [i]

                # fix non-noise model members to their respective models
                fixed = {}
                for model_id, model in models.iteritems():
                    fixed[model_id] = []
                    for group_id, group_members in tied.iteritems():
                        if len(set(group_members).intersection(
                                   set(assignments[model_id]))) > 0:
                            fixed[model_id].append(group_id)

                fixed['noise'] = []

                # perform clustering
                print 'Learning: ', collection_id
                models, assignments, c = cluster(models=models,
                                                 sequences=sequences,
                                                 assignments=assignments,
                                                 labels=labels,
                                                 fixed=fixed, tied=tied,
                                                 algorithm='baum-welch',
                                                 odir=odir)
            except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()

if __name__ == "__main__":
    alg = sys.argv[1]
    fixed = sys.argv[2]
    print alg
    if alg == 'viterbi' and fixed == 'fixed':
        kegg_init_fixed_vit()
    if alg == 'viterbi' and not fixed == 'fixed':
        kegg_init_vit()
    if alg == 'baum-welch' and fixed == 'fixed':
        kegg_init_fixed()
    if alg == 'baum-welch' and not fixed == 'fixed':
        kegg_init()

"""
odir_base = 'pathway_init_fixed_vit'  # directory to save files to
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
                                         odir=odir)"""
"""
        try:
            pathways, metab_pathway_names, gene_pathway_names, \
             unique_metab_pathway_members, unique_gene_pathway_members = \
             load_kegg_pathways(gc, mt, z)

            k = len(pathways)

            # number of pathways = number of models
            # metabolites/genes are assigned to a cluster pathway if they are
            # unique to that pathway. otherwise assign to noise
            assignments = defaultdict(list)
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

        except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            print >> f, "Unexpected error:", sys.exc_info()[0]
            f.close()
        """

"""
odir_base = 'pathway_init_unfixed_vit'  # directory to save files to
for n in state_range:
    for z in z_range:
        try:
            pathways, metab_pathway_names, gene_pathway_names, \
             unique_metab_pathway_members, unique_gene_pathway_members = \
             load_kegg_pathways(gc, mt, z)

            k = len(pathways)

            # number of pathways = number of models
            # metabolites/genes are assigned to a cluster pathway if they are
            # unique to that pathway. otherwise assign to noise
            assignments = defaultdict(list)
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

        except:
            error_file = odir.split('/') + ['errors.txt']
            error_file = '/'.join(error_file)
            f = open(error_file, 'a')
            print >> f, 'error computing parameters for: ', collection_id
            f.close()"""
