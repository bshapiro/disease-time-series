from config import config
from representation import Representation
from helpers import *
import matplotlib.pyplot as plt
from pickle import load, dump
from numpy import transpose, ndarray
from scipy.io import savemat
from sklearn.preprocessing import scale
from random import sample
from GP import fit_gp, fit_gp_with_priors

hRSV_matrix = transpose(load(open(config['hRSV_matrix'])))
h_sapiens_matrix = transpose(load(open(config['h_sapiens_matrix'])))
num_timepoints = h_sapiens_matrix.shape[1]
num_genes = h_sapiens_matrix.shape[0]

"""
scaled = scale(h_sapiens_matrix)
samples = sample(range(5000), 5)
for i in samples:
    plt.plot(scaled[i,:])
axes = plt.gca()
axes.set_ylim([-0.25, 0.25])
plt.show()
import pdb; pdb.set_trace()
"""

# create matrix of differences between times
d_matrix1 = ndarray((num_genes, 7))
for i in range(1, num_timepoints):
    d_matrix1[:, i - 1] = h_sapiens_matrix[:, i] - h_sapiens_matrix[:, i-1]

d_matrix2 = ndarray((10, 6))
for i in range(1, num_timepoints - 1):
    d_matrix2[:, i-1] = hRSV_matrix[:, i] - hRSV_matrix[:, i-1]

"""
scaled = scale(d_matrix2)
samples = range(10)
for i in samples:
    plt.plot(scaled[i,:])
plt.show()
import pdb; pdb.set_trace()
"""

d_svd = Representation(h_sapiens_matrix, 'svd', axis=0).getRepresentation()
U = d_svd[0]
S_vector = d_svd[1]
V = d_svd[2]
S = np.zeros((8, 8), dtype=float)
S[:8, :8] = np.diag(S_vector)
svd_loadings = np.dot(U, S)  # express each gene as a set of loadings on 7 canonical time series

# d_cca = Representation(h_sapiens_matrix[:, 1:], 'cca', axis=0, data2=hRSV_matrix).getRepresentation()
# cca_loadings = d_cca[0]
cca_loadings = svd_loadings  # TODO: remove

clusters1, cluster_centers1 = Representation(svd_loadings, 'kmeans', scale=False).getRepresentation()
clusters2, cluster_centers2 = Representation(cca_loadings, 'kmeans', scale=False).getRepresentation()

genes = load(open('../data/h_sapiens_genes.dump'))
cluster1_dict = {}
cluster2_dict = {}
for i in range(len(clusters1)):
    cluster_number1 = "c" + str(clusters1[i])
    cluster_number2 = "c" + str(clusters2[i])

    gene = genes[i]
    if cluster1_dict.get(cluster_number1) is None:
        cluster1_dict[cluster_number1] = []
    if cluster2_dict.get(cluster_number2) is None:
        cluster2_dict[cluster_number2] = []

    cluster1_dict[cluster_number1].append(gene)
    cluster2_dict[cluster_number2].append(gene)

for key, value in cluster1_dict.items():
    cluster1_dict[key] = np.asarray(value, dtype=np.object)
for key, value in cluster2_dict.items():
    cluster2_dict[key] = np.asarray(value, dtype=np.object)


savemat(open('../data/clusters1.mat', 'w'), cluster1_dict)
savemat(open('../data/clusters2.mat', 'w'), cluster2_dict)
savemat(open('../data/total_genes.mat', 'w'), {'genes': genes})

cluster1_time_series = {}
cluster2_time_series = {}

# scaled_h_sapiens_matrix = scale(h_sapiens_matrix)  # TODO: center appropriately? or don't center? 
scaled_h_sapiens_matrix = h_sapiens_matrix


for cluster_name, cluster_genes in cluster2_dict.items():
    sum_vector = np.zeros((8))
    for gene in cluster_genes:
        index = genes.index(gene)
        sum_vector += scaled_h_sapiens_matrix[index, :]
    cluster2_time_series[cluster_name] = sum_vector / len(cluster_genes)

for cluster_name, trajectory in cluster1_time_series.items():
    fit_gp_with_priors(trajectory[0], [0, 2, 4, 8, 12, 16, 20, 24])

"""
time = [0, 2, 4, 8, 12, 16, 20, 24]
plt.plot(time, component_0)
plt.plot(time, component_1)
plt.plot(time, component_2)
plt.show()
"""

# fig.savefig('hRSV_pca_c2.png')
#fig = plot_2D(hSRV_pca[0])
#plt.show()
