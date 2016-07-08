from config import param
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA, PCA
import sklearn.preprocessing as skp
from numpy.linalg import svd as svd_func
import numpy as np
from tools import rcca
from scipy.spatial.distance import cdist


class Representation:

    def __init__(self, view, view2=None, axis=0, scale=False):
        self.axis = axis
        self.view2 = None
        self.scale = scale
        if axis == 1:
            self.view = np.transpose(view)
            if view2 is not None:
                self.view2 = np.transpose(view2)
        else:
            self.view = view
            self.view2 = view2
        if self.scale:
            self.view = skp.scale(self.view)
            if self.view2 is not None:
                self.view2 = skp.scale(self.view2)

    def cca(self):
        print "Running CCA..."
        #model = CCA(n_components=config['cca_components'], scale=False)
        #model.fit(np.transpose(self.view), np.transpose(self.view2))
        #import pdb; pdb.set_trace()
        #return model.x_weights_, model.y_weights_, model.x_loadings_, model.y_loadings_
        model = rcca.CCA(kernelcca=False, numCC=param['components'], reg=param['cca_reg'])
        model.train([np.transpose(self.view), np.transpose(self.view2)])
        return model.ws[0], model.comps[0], model.ws[1], model.comps[1]

    def pca(self):
        print "Running PCA..."
        """
        Runs PCA on view and returns projected view, the principle components,
        and explained variance.
        """
        model = PCA()
        model.fit(self.view)
        return model.transform(self.view), model.components_, model.explained_variance_ratio_

    def svd(self):
        print "Running SVD..."
        """
        Returns (truncated) svd of the view.
        """
        svd = svd_func(self.view, full_matrices=False)
        return svd

    def sparse_pca(self):
        """
        Runs PCA on view and returns projected view, the principle components,
        and explained variance.
        """
        model = SparsePCA(n_components=param['components'], alpha=param['sparse_pca_alpha'])
        model.fit(self.view)
        return model.transform(self.view), model.components_

    def kmeans(self):
        print "Running kmeans..."
        km = KMeans(n_clusters=param['kmeans_clusters'])
        km.fit(self.view)

        return km.labels_, km.cluster_centers_
