from scipy.cluster import hierarchy
import sys
from load_data import load_data
from pickle import dump, load
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

genefile = '3k_genes.p'
gc, mt, track = load_data()
genes = load(open(genefile,  'r'))
data = gc.data.loc[genes, :]

directories = glob.glob('/'.join(sys.argv[1].split('/')))
print directories

for directory in directories:
    linkage_path = '/'.join(directory.split('/') + ['linkage.p'])
    cid = directory.split('/')[-1]
    out_directory = '/'.join(sys.argv[3].split('/') + [cid])

    Z = load(open(linkage_path, 'r'))

    hierarchy.dendrogram(Z)
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    dendrogram_file = '/'.join(out_directory.split('/') + ['dendrogram.png'])
    plt.savefig(dendrogram_file)

    k = int(sys.argv[2])
    fclust = hierarchy.fcluster(Z, k, 'maxclust')
    fclust -= 1
    clustering = {}
    for i in range(0, k):
        clustering[i] = []
        for j in np.where(fclust == i)[0]:
            clustering[i] = clustering[i] + [data.index[j]]

    # write cluster assignments
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    filepath = '/'.join(out_directory.split('/') + ['assignments.p'])
    dump(clustering, open(filepath, 'w'))

    filepath = out_directory.split('/') + ['assignments.txt']
    filepath = '/'.join(filepath)
    with open(filepath, 'w') as f:
        for cluster, members in clustering.iteritems():
            f.write(str(cluster))
            f.write('\n')
            f.write('\t'.join(members))
            f.write('\n')
            f.write('\n')
            f.write('\n')
