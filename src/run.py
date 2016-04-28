from config import config
from representation import Representation
from helpers import *
import matplotlib.pyplot as plt
from pickle import load

hRSV_matrix = load(open(config['hRSV_matrix']))
h_sapiens_matrix = load(open(config['h_sapiens_matrix']))
h_sapiens_pca = Representation(h_sapiens_matrix, 'sparse_pca', 0).getRepresentation()
hRSV_pca = Representation(hRSV_matrix, 'sparse_pca', 0).getRepresentation()
import pdb; pdb.set_trace()
fig = plot_2D(hRSV_pca[0])
plt.show()
fig.savefig('hRSV_pca_c2.png')
#fig = plot_2D(hSRV_pca[0])
#plt.show()
