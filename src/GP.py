from config import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import GPy
import numpy as np
import pdb
import pylab as pl
import sys
pl.ion()


def process_data(filename):
    data = np.genfromtxt(filename, comments='!', delimiter='\t')
    data = data[1::, 1::]  # Drop first row and column of label data since it just becomes NaN
    p, t = data.shape  # t for number of timepoints, p fro number of proteins
    return data


def fit_gp(component):
    # fit a gp to the time series data for this component
    if config['gp_type'] == 'rbf':
        kernel = GPy.kern.RBF(input_dim=1, variance=config['gp_variance'], lengthscale=1.)
    elif config['gp_type'] == 'exp':
        kernel = GPy.kern.Exponential(input_dim=1, variance=config['gp_variance'], lengthscale=1.)
    else:
        sys.exit('No kernel specified.')

    x = np.reshape(np.asfarray(range(len(component))), (len(component), 1))
    y = np.reshape(component, (len(component), 1))

    y_mean = np.mean(y)
    y_centered = y - y_mean

    m = GPy.models.GPRegression(x, y_centered, kernel)
    print m
    m.plot()
    pdb.set_trace()


def pca(data):
    pca = PCA(n_components=config['pca_components'])
    pca.fit(np.transpose(data))
    components = pca.components_
    projected_data = np.dot(components, data)
    return projected_data


def learn_representation(data):
    if config['representation'] == 'pca':
        return pca(data)
    elif config['representation'] == 'raw':
        return data


if __name__ == "__main__":
    data = process_data(config['autoantibody_filename'])
    data = scale(data, 1)
    representation = learn_representation(data)
    gps = []
    for i in range(np.shape(representation)[0]):
        component = representation[i]
        gps.append(fit_gp(component))
