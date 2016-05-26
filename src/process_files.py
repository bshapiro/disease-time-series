from glob import glob
from numpy import loadtxt, ndarray, asarray, transpose
import pickle

# hRSV_files = glob("../data/*hRSV.mRNA.*")
# h_sapiens_files = glob("../data/*h_sapiens.mRNA.*")

# print hRSV_files
# print h_sapiens_files


# hRSV_matrix = ndarray((7, 10))

# for i in range(len(hRSV_files)):
#     f = hRSV_files[i]
#     vector = loadtxt(f, usecols=[1])
#     hRSV_matrix[i] = vector

# f = open(f)
# hRSV_genes = [line.split()[0] for line in f.readlines()[:-1]]
# print hRSV_genes
# f.close()


# h_sapiens_matrix = ndarray((8, 51055))

# for i in range(len(h_sapiens_files)):
#     f = h_sapiens_files[i]
#     vector = loadtxt(f, usecols=[1])
#     h_sapiens_matrix[i] = vector

# f = open(f)
# h_sapiens_genes = [line.split()[0] for line in f.readlines()[:-1]]
# print h_sapiens_genes

# pickle.dump(h_sapiens_matrix, open('../data/h_sapiens_matrix.dump', 'w'))
# pickle.dump(hRSV_matrix, open('../data/hRSV_matrix.dump', 'w'))

usc_genes = [gene[:-1] for gene in open('../data/myeloma/usc_genes.txt').readlines()]
map_lines = open('../data/myeloma/myeloma_genes.txt').readlines()
gene_map = {}
for line in map_lines:
    line = line[:-1].split(',')
    gene_map[line[1]] = line[0]
polyA = pickle.load(open('../data/myeloma/polyA.dump'))
ribosome = pickle.load(open('../data/myeloma/ribosome.dump'))
new_polyA = []
new_ribosome = []
new_genes = []
for i in range(polyA.shape[1]):
    usc_gene = usc_genes[i]
    gene_name = gene_map.get(usc_gene)
    if gene_name is not None:
        new_polyA.append(polyA[:, i])
        new_ribosome.append(ribosome[:, i])
        new_genes.append(gene_name)


new_polyA = transpose(asarray(new_polyA))
new_ribosome = transpose(asarray(new_ribosome))

pickle.dump(new_polyA, open('../data/myeloma/polyA.dump', 'w'))
pickle.dump(new_ribosome, open('../data/myeloma/ribosome.dump', 'w'))
pickle.dump(new_genes, open('../data/myeloma/myeloma_genes.dump', 'w'))
