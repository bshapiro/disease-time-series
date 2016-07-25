import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import seaborn as sns

"""
point to a directory with subfolders where each subfolder is a clustering
with a gene and metabolite set enrichment results subfolder
generate a histogram of enrichment counts
"""


def run(cluster_directory_root):
    cluster_directories = \
        glob.glob(cluster_directory_root + '/*/')

    clustering_ids = []
    genriched_clusters = []
    genrichments_per_cluster = []
    menriched_clusters = []
    menrichments_per_cluster = []
    for cluster_dir in cluster_directories:
        try:
            gene_file = '/'.join(cluster_dir.split('/') +
                                 ['gene_enrichment', 'summary'])
            metab_file = '/'.join(cluster_dir.split('/') +
                                  ['metab_enrichment', 'summary'])
            clustering_id = cluster_dir.split('/')[-2:][0]
            print clustering_id

            glines = (open(gene_file, 'r').read().splitlines())
            mlines = (open(metab_file, 'r').read().splitlines())

            genriched = 0.1
            genriched_count = 0.1
            menriched = 0.1
            menriched_count = 0.1

            for i, line in enumerate(glines):
                if i == 0:
                    continue
                if len(line) > 0 and line[0] != '\t':
                    genriched += 1
                elif len(line) > 0 and line[0] == '\t':
                    genriched_count += 1

            for i, line in enumerate(mlines):
                if i == 0:
                    continue
                if len(line) > 0 and line[0] != '\t':
                    menriched += 1
                elif len(line) > 0 and line[0] == '\t':
                    menriched_count += 1

            genriched_clusters.append(genriched)
            genrichments_per_cluster.append(genriched_count)
            menriched_clusters.append(menriched)
            menrichments_per_cluster.append(menriched_count)
            # clustering_ids.append(clustering_id.split('_')[1])
            clustering_ids.append(clustering_id)
        except:
            pass

    df = pd.DataFrame([pd.Series(index=clustering_ids, data=genriched_clusters, name='Gene Enriched Clusters'),
                       pd.Series(index=clustering_ids, data=genrichments_per_cluster, name='Gene Enriched Pathways'),
                       pd.Series(index=clustering_ids, data=menriched_clusters, name='Metabolite Enirched Clusters'),
                       pd.Series(index=clustering_ids, data=menrichments_per_cluster, name='Metabolite Enirched Pathways')
                       ])
    df.T.plot.bar()
    plt.title('Enrichment Counts by Clustering')
    savename = '/'.join(cluster_directory_root.split('/') +
                        ['enrichment_counts'])
    plt.savefig(savename)
    plt.close()
    print 'Saved plot to: ', savename

if __name__ == "__main__":
    directory = sys.argv[1]
    print 'genes'
    print directory
    run(directory)
