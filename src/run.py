from preprocessing import *

mc_data = '../data/my_connectome/GSE58122_varstab_data_prefiltered_geo.txt'
mc_genes = '../data/my_connectome/mc_geneset.p'
fd = ['../data/my_connectome/rin.p']
ops = ['>']
th = [6]

p0 = Preprocessing(mc_data, filter_data=fd, operators=ops, thresholds=th, clean_components=None, sample_labels=mc_genes)
p1 = Preprocessing(mc_data, filter_data=fd, operators=ops, thresholds=th, clean_components=[0], sample_labels=mc_genes)
p2 = Preprocessing(mc_data, filter_data=fd, operators=ops, thresholds=th, clean_components=[0,1], sample_labels=mc_genes)
p3 = Preprocessing(mc_data, filter_data=fd, operators=ops, thresholds=th, clean_components=[0,1,2], sample_labels=mc_genes)

pca = PCA()

print pca.fit(p0.data).explained_variance_ratio_
print pca.fit(p1.data).explained_variance_ratio_
print pca.fit(p2.data).explained_variance_ratio_
print pca.fit(p3.data).explained_variance_ratio_