from config import config
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA, PCA
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as skp
from numpy.linalg import svd as svd_func
from collections import defaultdict
import numpy as np
import itertools
from tools import rcca


class Representation:

    def __init__(self, data, representation_type, data2=None, axis=0, scale=True):
        if representation_type is not None:
            self.method = getattr(self, representation_type)
        self.axis = axis
        self.data2 = None
        self.scale = scale
        if axis == 1:
            self.data = np.transpose(data)
            if data2 is not None:
                self.data2 = np.transpose(data2)
        else:
            self.data = data
            self.data2 = data2
        if self.scale:
            self.data = skp.scale(self.data)
            if self.data2 is not None:
                self.data2 = skp.scale(self.data2)

    def cca(self):
        print "Running CCA..."
        #model = CCA(n_components=config['cca_components'], scale=False)
        #model.fit(np.transpose(self.data), np.transpose(self.data2))
        #import pdb; pdb.set_trace()
        #return model.x_weights_, model.y_weights_, model.x_loadings_, model.y_loadings_
        model = rcca.CCA(kernelcca=False, numCC=config['cca_components'], reg=config['cca_reg'])
        model.train([np.transpose(self.data), np.transpose(self.data2)])
        return model.ws[0], model.comps[0], model.ws[1], model.comps[1]

    def pca(self):
        print "Running PCA..."
        """
        Runs PCA on data and returns projected data, the principle components,
        and explained variance.
        """
        model = PCA()
        model.fit(self.data)
        return model.transform(self.data), model.components_, model.explained_variance_ratio_

    def svd(self):
        print "Running SVD..."
        """
        Returns (truncated) svd of the data.
        """
        svd = svd_func(self.data, full_matrices=False)
        return svd

    def sparse_pca(self):
        """
        Runs PCA on data and returns projected data, the principle components,
        and explained variance.
        """
        model = SparsePCA(n_components=config['pca_components'], alpha=config['sparse_pca_alpha'])
        model.fit(self.data)
        return model.transform(self.data), model.components_

    def kmeans(self):
        print "Running kmeans..."
        
        conservation = []
        if type(config['kmeans_clusters']) is list:
            # if its a list we are trying to generate a comparison of cluster sizes
            # we compare by looking at pair-wise cluster assignment across 10 iterations
            cluster_sizes = config['kmeans_clusters']
            for k in cluster_sizes:
                print 'k= :', k
                index_pairs = defaultdict(int)
                for repeat in range(2):
                    km = KMeans(n_clusters=k)
                    km.fit(self.data)
                    l = km.labels_
                    for i in range(k):
                        pairs = set(itertools.combinations(np.where(l==i)[0].tolist(), 2))
                        for pair in pairs:
                            index_pairs.__getitem__(pair)
                            index_pairs[pair] += 1
                conservation.append(sum(index_pairs.values()) / float((len(index_pairs.keys()) * 2)))
            print conservation
                
        
        km = KMeans(n_clusters=config['kmeans_clusters'])
        km.fit(self.data)

        return km.labels_, km.cluster_centers_, conservation
    """
    def kmeans(self, data, n=config['kmeans_clusters'], axis=0):

            model = KMeans(n_clusters=n)
        model.fit(data)
        rep = np.zeros((n, model.labels_.size))

        if config['kmeans_representation'] == 'binary':
            for i in range (0, model.labels_.size):
                rep[model.labels_[i], i] = 1

        if config['kmeans_representation'] == 'distance':
            rep = cdist(model.cluster_centers_, data, 'euclidean')

        return rep, model.labels_, model.cluster_centers_
    """

    def getRepresentation(self):
        return self.method()
