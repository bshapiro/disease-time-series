from scipy.cluster import hierarchy
import sys
from load_data import load_data
from pickle import dump, load
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os

genefile = '1k_genes.p'
gc, mt, track = load_data()
genes = load(open(genefile,  'r'))
data = gc.data.loc[genes, :]

linkage_path = '/'.join(sys.argv[1].split('/') + ['linkage'])
Z = load(open(linkage_path, 'r'))

hierarchy.dendrogram(Z)
dendrogram_file = '/'.join(sys.argv[1].split('/') + ['dendrogram.png'])
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
out_directory = '/'.join(sys.argv[1].split('/'))
if not os.path.isdir(out_directory):
    os.makedirs(out_directory)
filepath = out_directory.split('/') + ['agglomerative_clustering_assignments.txt']
filepath = '/'.join(filepath)
with open(filepath, 'w') as f:
    for cluster, members in clustering.iteritems():
        f.write(str(cluster))
        f.write('\n')
        f.write('\t'.join(members))
        f.write('\n')
        f.write('\n')
        f.write('\n')