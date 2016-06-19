from config import config, param
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA, PCA
import sklearn.preprocessing as skp
from numpy.linalg import svd as svd_func
import numpy as np
from tools import rcca
from scipy.spatial.distance import cdist


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
        model = rcca.CCA(kernelcca=False, numCC=param['components'], reg=param['cca_reg'])
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
        model = SparsePCA(n_components=param['components'], alpha=param['sparse_pca_alpha'])
        model.fit(self.data)
        return model.transform(self.data), model.components_

    def kmeans(self):
        print "Running kmeans..."
        km = KMeans(n_clusters=param['kmeans_clusters'])
        km.fit(self.data)

        return km.labels_, km.cluster_centers_

    def kmeans_karl(self, data, n=param['kmeans_clusters'], axis=0):
        """
        Kmeans clustering on data, each row of data is a sample, columns are features
        returns representation of data from kmeans results (set in config)
            -binary = 1/0 for cluster assignments
            -distance = euclidean distance to each cluster
        array of cluster labels [0, k-1], and matrix of cluster centers
        """

        model = KMeans(n_clusters=n)
        model.fit(data)
        rep = np.zeros((n, model.labels_.size))

        if param['kmeans_representation'] == 'binary':
            for i in range(0, model.labels_.size):
                rep[model.labels_[i], i] = 1

        if param['kmeans_representation'] == 'distance':
            rep = cdist(model.cluster_centers_, data, 'euclidean')

        return rep, model.labels_, model.cluster_centers_

    def getRepresentation(self):
        return self.method()
