import numpy as np
import os


def make_set(save_name, sets, set_index):
    try:
        f = open(save_name, 'w')
    except:
        directory = '/'.join(save_name.split('/')[:-1])
        print "Creating directory...", directory
        os.mkdir(directory)
        f = open(save_name, 'w')

    for i in range(sets.shape[0]):
        line = sets[i][0] + '\t' + sets[i][1] + '\t' + \
               sets[i][set_index].replace(',', '\t')
        print >> f, line


geneset_filepath = 'CPDB_pathways_genes.tab'
genesets = (open(geneset_filepath, 'r').read().splitlines())

genesets = genesets[1:]  # remove header line
genesets = [geneset.split('\t') for geneset in genesets]
genesets = np.array(genesets)

metabset_filepath = 'CPDB_pathways_metabolites.tab'
metabsets = (open(metabset_filepath, 'r').read().splitlines())

metabsets = metabsets[1:]  # remove header line
metabsets = [metabset.split('\t') for metabset in metabsets]
metabsets = np.array(metabsets)
for i in range(metabsets.shape[0]):
    metabsets[i, 2] = metabsets[i, 2].replace('kegg:', '')

kegg_genesets = genesets[np.where(genesets[:, 2] == 'KEGG')]
kegg_metabsets = metabsets[np.where(metabsets[:, 1] == 'KEGG')]

reactome_genesets = genesets[np.where(genesets[:, 2] == 'Reactome')]
reactome_metabsets = metabsets[np.where(metabsets[:, 1] == 'Reactome')]

save_name = 'KEGG_genes'
sets = kegg_genesets
make_set(save_name, sets, 3)

save_name = 'KEGG_metabolites'
sets = kegg_metabsets
make_set(save_name, sets, 2)

save_name = 'REACTOME_genes'
sets = reactome_genesets
make_set(save_name, sets, 3)

save_name = 'REACTOME_metabolites'
sets = reactome_metabsets
make_set(save_name, sets, 2)

save_name = 'CPDB_genes'
sets = genesets
make_set(save_name, sets, 3)

save_name = 'CPDB_metabolites'
sets = metabsets
make_set(save_name, sets, 2)
