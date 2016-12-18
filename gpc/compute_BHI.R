# install.packages('clValid')
library('clValid')
clusters_filename = '~/Documents/Research/Projects/disease-time-series/gpc/single/kmeans/avg/raw/clusters5/polya/memberships.txt'
genes_filename = '~/Documents/Research/Projects/disease-time-series/data/myeloma/genes.csv'
cluster_memberships = as.integer(c(read.table(clusters_filename, sep=','))) + 1;
gene_names = as.vector(read.table(genes_filename, sep=',', stringsAsFactors=F))$V1;
names(cluster_memberships) <- gene_names
BHI(cluster_memberships, 'hgu133plus2.db', names=gene_names, category="all");
