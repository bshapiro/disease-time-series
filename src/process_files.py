from glob import glob
from numpy import loadtxt, ndarray
import pickle

hRSV_files = glob("../data/*hRSV.mRNA.*")
h_sapiens_files = glob("../data/*h_sapiens.mRNA.*")

print hRSV_files
print h_sapiens_files


hRSV_matrix = ndarray((7, 10))

for i in range(len(hRSV_files)):
    f = hRSV_files[i]
    vector = loadtxt(f, usecols=[1])
    hRSV_matrix[i] = vector

f = open(f)
hRSV_genes = [line.split()[0] for line in f.readlines()[:-1]]
print hRSV_genes
f.close()


h_sapiens_matrix = ndarray((8, 51055))

for i in range(len(h_sapiens_files)):
    f = h_sapiens_files[i]
    vector = loadtxt(f, usecols=[1])
    h_sapiens_matrix[i] = vector

f = open(f)
h_sapiens_genes = [line.split()[0] for line in f.readlines()[:-1]]
print h_sapiens_genes

pickle.dump(h_sapiens_matrix, open('../data/h_sapiens_matrix.dump', 'w'))
pickle.dump(hRSV_matrix, open('../data/hRSV_matrix.dump', 'w'))