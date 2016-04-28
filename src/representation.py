from config import config
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA, PCA
import numpy as np


class Representation:

    def __init__(self, data, representation_type, axis=0):
        method = getattr(self, representation_type)
        self.axis = axis
        if axis == 1:
            self.data = np.transpose(data)
        else:
            self.data = data
        self.representation = method()

    def pca(self):
        """
        Runs PCA on data and returns projected data, the principle components,
        and explained variance.
        """
        model = PCA(n_components=config['pca_components'])
        model.fit(self.data)
        return model.transform(self.data), model.components_, model.explained_variance_ratio_

    def sparse_pca(self):
        """
        Runs PCA on data and returns projected data, the principle components,
        and explained variance.
        """
        model = SparsePCA(n_components=config['pca_components'], alpha=10)
        model.fit(self.data)
        return model.transform(self.data), model.components_, model.explained_variance_ratio_

    def kmeans(self):
        km = KMeans(n_clusters=config['kmeans_clusters'])
        km.fit(self.data)
        return km.labels_, km.cluster_centers_

    def getRepresentation(self):
        return self.representation
