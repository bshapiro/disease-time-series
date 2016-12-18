import addpath
from pickle import load
from src.gsea.run_enrichment import run_enrichment
from src.gsea.process_gsea_output import get_sigs
from src.tools.cluster_evaluation import (clusterings_conserved_pairs,
                                          clusterwise_conserved_pairs)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import sys
import os

from collections import defaultdict
import re

kegg_file = '../src/gsea/KEGG_genes'
reactome_file = '../src/gsea/REACTOME_genes'
GO_BP_file = '../src/gsea/GO-BP.gmt'
GO_MF_file = '../src/gsea/GO-MF.gmt'
GO_CC_file = '../src/gsea/GO-CC.gmt'
pathways = {'KEGG': kegg_file, 'REACTOME': reactome_file, 'GO-BP': GO_BP_file,
            'GO-MF': GO_MF_file, 'GO_CC': GO_CC_file}
pathways = {'KEGG': kegg_file, 'REACTOME': reactome_file, 'GO-BP': GO_BP_file}


dirs = sys.argv[1:]
clusterings = {}
genes = []

for d in dirs:
    cluster_directories = \
    glob.glob(d + '/*/')
    print cluster_directories
    for cluster_dir in cluster_directories:
            # read final clusters
            filepath = '/'.join(cluster_dir.split('/') + ['assignments.p'])
            clusters = load(open(filepath, 'r'))
            print clusters
            genes = set()
            for cid, members in clusters.iteritems():
                genes.update(list(set(members)))

            for pathway, pathway_file in pathways.iteritems():
                directory = cluster_dir + '/' + pathway + '/'
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                run_enrichment(clusters, genes, pathway_file, directory)

                sig_genes = get_sigs(directory)
                with(open(cluster_dir + '/' + pathway + 'summary', 'w')) as f:
                    print >> f,  "Significant enrichments:"
                    for key, value in sig_genes.items():
                        print >> f, key
                        for sig in value:
                            print >> f, '\t', sig[0], sig[-3], sig[-2], sig[-1]
