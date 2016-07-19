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
    gsequences, glabels = df_to_sequence_list(gc.data)

    sequences = np.concatenate((msequences, gsequences), 0)
    labels = np.concatenate((mlabels, glabels))

    sequences = np.concatenate((sequences, -1 * sequences))

    # tie positive and negative expression sequences
    tied = {}
    for i, label in enumerate(labels):
        tied[label] = [i, i+labels.size]

    state_labels = np.concatenate(((labels + '+'), (labels + '-')))
    labels = np.concatenate((labels, labels))

    # noise model trained on all data once
    # genes/metabolites will be assigned to noise model if other models
    # fail to model it better
    noise_dist = [NormalDistribution(0, 1)]
    noise_trans = np.array([[1]])
    starts = np.array([1])
    noise = HiddenMarkovModel.from_matrix(noise_trans, noise_dist, starts)
    noise.freeze_distributions()
    return gc, mt, sequences, labels, state_labels, tied, noise, z_range, \
        state_range


def kegg_init_invert_vit():
    gc, mt, sequences, labels, state_labels, tied, noise, z_range, \
        state_range = init()

    for n in state_range:
        for z in z_range:
            try:
                pathways, metab_pathway_names, gene_pathway_names, \
                 unique_metab_pathway_members, unique_gene_pathway_members = \
                 load_kegg_pathways(gc, mt, z)

                k = len(pathways)
                # directory to save files to
                odir_base = '../results/khmm/viterbi/kegg_init_invert'
                collection_id = 'k-' + str(k) + '_z-' + str(z) + '_n-' \
                    + str(n) + '_kegg_init_invert_vit'
                odir = odir_base + '/' + collection_id

                # number of pathways = number of models
                # metabolites/genes are assigned to a cluster pathway if
                # unique to that pathway. otherwise assign to noise
                assignments = defaultdict(list)
                for group_id, group_members in tied.iteritems():
                    placed = False
                    for j, pathway in enumerate(pathways):
                        # if metabolite is unique to cluster, assign
                        if pathway in metab_pathway_names:
                            index = metab_pathway_names.index(pathway)
                            if group_id in unique_metab_pathway_members[index]:
                                assignments[pathway].append(group_members[0])
                                placed = True
                        # if gene is unique to cluster, assign
                        if pathway in gene_pathway_names:
                            index = gene_pathway_names.index(pathway)
                            if group_id in unique_gene_pathway_members[index]:
                                assignments[pathway].append(group_members[0])
                                placed = True
                    # assign to noise otherwise
                    if not placed:
                        assignments['noise'].append(group_members[0])

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
                                                 state_labels=state_labels,
                                                 tied=tied,
                                                 algorithm='viterbi',
                                                 odir=odir)
            except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()


def kegg_init_invert():
    gc, mt, sequences, labels, state_labels, tied, noise, z_range, \
        state_range = init()

    for n in state_range:
        for z in z_range:
            try:
                pathways, metab_pathway_names, gene_pathway_names, \
                 unique_metab_pathway_members, unique_gene_pathway_members = \
                 load_kegg_pathways(gc, mt, z)

                k = len(pathways)
                # directory to save files to
                odir_base = '../results/khmm/kegg_init_invert'
                collection_id = 'k-' + str(k) + '_z-' + str(z) + '_n-' \
                    + str(n) + '_kegg_init_invert_vit'
                odir = odir_base + '/' + collection_id

                # number of pathways = number of models
                # metabolites/genes are assigned to a cluster pathway if
                # unique to that pathway. otherwise assign to noise
                assignments = defaultdict(list)
                for group_id, group_members in tied.iteritems():
                    placed = False
                    for j, pathway in enumerate(pathways):
                        # if metabolite is unique to cluster, assign
                        if pathway in metab_pathway_names:
                            index = metab_pathway_names.index(pathway)
                            if group_id in unique_metab_pathway_members[index]:
                                assignments[pathway].append(group_members[0])
                                placed = True
                        # if gene is unique to cluster, assign
                        if pathway in gene_pathway_names:
                            index = gene_pathway_names.index(pathway)
                            if group_id in unique_gene_pathway_members[index]:
                                assignments[pathway].append(group_members[0])
                                placed = True
                    # assign to noise otherwise
                    if not placed:
                        assignments['noise'].append(group_members[0])

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
                                                 state_labels=state_labels,
                                                 tied=tied,
                                                 algorithm='baum-welch',
                                                 odir=odir)
            except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()


def kegg_init_invert_fixed_vit():
    gc, mt, sequences, labels, state_labels, tied, noise, z_range, \
        state_range = init()

    for n in state_range:
        for z in z_range:
            try:
                pathways, metab_pathway_names, gene_pathway_names, \
                 unique_metab_pathway_members, unique_gene_pathway_members = \
                 load_kegg_pathways(gc, mt, z)

                k = len(pathways)
                # directory to save files to
                odir_base = '../results/khmm/viterbi/kegg_init_invert_fixed'
                collection_id = 'k-' + str(k) + '_z-' + str(z) + '_n-' \
                    + str(n) + '_kegg_init_invert_vit'
                odir = odir_base + '/' + collection_id

                # number of pathways = number of models
                # metabolites/genes are assigned to a cluster pathway if
                # unique to that pathway. otherwise assign to noise
                assignments = defaultdict(list)
                for group_id, group_members in tied.iteritems():
                    placed = False
                    for j, pathway in enumerate(pathways):
                        # if metabolite is unique to cluster, assign
                        if pathway in metab_pathway_names:
                            index = metab_pathway_names.index(pathway)
                            if group_id in unique_metab_pathway_members[index]:
                                assignments[pathway].append(group_members[0])
                                placed = True
                        # if gene is unique to cluster, assign
                        if pathway in gene_pathway_names:
                            index = gene_pathway_names.index(pathway)
                            if group_id in unique_gene_pathway_members[index]:
                                assignments[pathway].append(group_members[0])
                                placed = True
                    # assign to noise otherwise
                    if not placed:
                        assignments['noise'].append(group_members[0])

                # create model
                models = defaultdict(list)
                for pathway, seq_indices in assignments.iteritems():
                    model = init_gaussian_hmm(sequences[seq_indices, :], n,
                                              model_id=pathway)
                    models[pathway] = model

                # add noise model
                models['noise'] = noise

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
                                                 state_labels=state_labels,
                                                 tied=tied, fixed=fixed,
                                                 algorithm='viterbi',
                                                 odir=odir)
            except:
                error_file = odir.split('/') + ['errors.txt']
                error_file = '/'.join(error_file)
                f = open(error_file, 'a')
                print >> f, 'error computing parameters for: ', collection_id
                print >> f, "Unexpected error:", sys.exc_info()[0]
                f.close()

def kegg_init_invert_fixed():
    gc, mt, sequences, labels, state_labels, tied, noise, z_range, \
        state_range = init()

    for n in state_range:
        for z in z_range:
            try:
                pathways, metab_pathway_names, gene_pathway_names, \
                 unique_metab_pathway_members, unique_gene_pathway_members = \
                 load_kegg_pathways(gc, mt, z)

                k = len(pathways)
                # directory to save files to
                odir_base = '../results/khmm/kegg_init_invert_fixed'
                collection_id = 'k-' + str(k) + '_z-' + str(z) + '_n-' \
                    + str(n) + '_kegg_init_invert_vit'
                odir = odir_base + '/' + collection_id

                # number of pathways = number of models
                # metabolites/genes are assigned to a cluster pathway if
                # unique to that pathway. otherwise assign to noise
                assignments = defaultdict(list)
                for group_id, group_members in tied.iteritems():
                    placed = False
                    for j, pathway in enumerate(pathways):
                        # if metabolite is unique to cluster, assign
                        if pathway in metab_pathway_names:
                            index = metab_pathway_names.index(pathway)
                            if group_id in unique_metab_pathway_members[index]:
                                assignments[pathway].append(group_members[0])
                                placed = True
                        # if gene is unique to cluster, assign
                        if pathway in gene_pathway_names:
                            index = gene_pathway_names.index(pathway)
                            if group_id in unique_gene_pathway_members[index]:
                                assignments[pathway].append(group_members[0])
                                placed = True
                    # assign to noise otherwise
                    if not placed:
                        assignments['noise'].append(group_members[0])

                # create model
                models = defaultdict(list)
                for pathway, seq_indices in assignments.iteritems():
                    model = init_gaussian_hmm(sequences[seq_indices, :], n,
                                              model_id=pathway)
                    models[pathway] = model

                # add noise model
                models['noise'] = noise

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
                                                 state_labels=state_labels,
                                                 tied=tied, fixed=fixed,
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
        kegg_init_invert_fixed_vit()
    if alg == 'viterbi' and not fixed == 'fixed':
        kegg_init_invert_vit()
    if alg == 'baum-welch' and fixed == 'fixed':
        kegg_init_invert_fixed()
    if alg == 'baum-welch' and not fixed == 'fixed':
        kegg_init_invert()
