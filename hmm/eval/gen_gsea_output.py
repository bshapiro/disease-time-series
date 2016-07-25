import addpath
import numpy as np
from src.gsea.run_enrichment import run_enrichment
from src.gsea.process_gsea_output import get_sigs
from hmm.load_data import load_data
import glob
import sys


def run(cluster_directory_root, pathways):
    gc, mt, track = load_data(None, 0)

    """
    background_genes = gc.data.index.values
    background_metabolites = mt.data.index.values
    background_joint = np.concatenate((background_genes,
                                       background_metabolites))
    """

    kegg_gene_file = \
        '../src/gsea/KEGG_genes'

    kegg_metab_file = \
        '../src/gsea/' + \
        'KEGG_metabolites'

    reactome_gene_file = \
        '../src/gsea/REACTOME_genes'

    reactome_metab_file = \
        '../src/gsea/' + \
        'REACTOME_metabolites'

    all_gene_file = \
        '../src/gsea/CPDB_genes'

    all_metab_file = \
        '../src/gsea/' + \
        'CPDB_metabolites'

    if pathways == 'kegg':
        gene_file = kegg_gene_file
        metab_file = kegg_metab_file
    if pathways == 'reactome':
        gene_file = reactome_gene_file
        metab_file = reactome_metab_file
    if pathways == 'all':
        gene_file = all_gene_file
        metab_file = all_metab_file

    cluster_directories = \
        glob.glob(cluster_directory_root + '/*/')

    print cluster_directories

    # read initial clusters
    initial_clusters = {}
    for cluster_dir in cluster_directories:
        try:
            filepath = '/'.join(cluster_dir.split('/') +
                                ['init_assignments.txt'])
            lines = (open(filepath, 'r').read().splitlines())
            l = 0
            while l < len(lines):
                cluster_name = lines[l]
                cluster_members = lines[l + 1].split('\t')
                # hack to fix output file mistake
                for i, member in enumerate(cluster_members):
                    if member[-1:] == '+' or member[-1:] == '-':
                        cluster_members[i] = member[:-1]
                initial_clusters[cluster_name] = cluster_members
                l += 4
        except:
            pass

    background_genes = set()
    background_metabolites = set()
    for cid, members in initial_clusters.iteritems():
        background_genes.update(list(set(members).
                                intersection(set(gc.data.index.values.
                                             tolist()))))
        background_metabolites.update(list(set(members).
                                      intersection(set(mt.data.index.values.
                                                   tolist()))))

    for cluster_dir in cluster_directories:
        try:
            # read final clusters
            joint_clusters = {}
            filepath = '/'.join(cluster_dir.split('/') + ['assignments.txt'])
            lines = (open(filepath, 'r').read().splitlines())
            l = 0
            while l < len(lines):
                cluster_name = lines[l]
                cluster_members = lines[l + 1].split('\t')
                # hack to fix output file mistake
                for i, member in enumerate(cluster_members):
                    if member[-1:] == '+' or member[-1:] == '-':
                        cluster_members[i] = member[:-1]
                joint_clusters[cluster_name] = cluster_members
                l += 4

            # seperate metabolite and gene cluster results
            metab_clusters = {}
            gene_clusters = {}
            for cluster, members in joint_clusters.iteritems():
                ms = set(members).intersection(background_metabolites)
                gs = set(members).intersection(background_genes)
                metab_clusters[cluster] = list(ms)
                gene_clusters[cluster] = list(gs)

            # run gene enrichment
            genedir = '/'.join(cluster_dir.split('/') + ['gene_enrichment'])
            genedir = genedir + '/'
            run_enrichment(gene_clusters, background_genes, gene_file,
                           genedir)

            # run metabolite enrichment
            metabdir = '/'.join(cluster_dir.split('/') + ['metab_enrichment'])
            metabdir = metabdir + '/'
            run_enrichment(metab_clusters, background_metabolites,
                           metab_file, metabdir)

            # summarize
            sig_genes = get_sigs(genedir)
            with(open(genedir + 'summary', 'w')) as f:
                print >> f,  "Significant enrichments:"
                for key, value in sig_genes.items():
                    print >> f, key
                    for sig in value:
                        print >> f, '\t', sig[0], sig[-3], sig[-2], sig[-1]

            sig_metabs = get_sigs(metabdir)
            with(open(metabdir + 'summary', 'w')) as f:
                print >> f,  "Significant enrichments:"
                for key, value in sig_metabs.items():
                    print >> f, key
                    for sig in value:
                        print >> f, '\t', sig[0], sig[-3], sig[-2], sig[-1]
        except:
            pass

if __name__ == "__main__":
    directory = sys.argv[1]
    pathways = sys.argv[2]
    print directory, pathways
    run(directory, pathways)
