import addpath
from src.khmm import *
from src.tools.helpers import *
import numpy as np
from pickle import load, dump
from hmmlearn import hmm, utils
from load_data import gc, mt

def load_kegg_pathways(z):
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

    return pathways, metab_pathway_names, gene_pathway_names, \
        unique_metab_pathway_members, unique_gene_pathway_members
