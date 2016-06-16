import pandas as pd
import numpy as np
import sys
import os
from pickle import load, dump
from config import config
from optparse import OptionParser
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import *
from tools.helpers import *
from representation import *

parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset", default=None,
                  help="Dataset")
parser.add_option("--dir", dest="directory",
                  help="Directory to look for data files.")
parser.add_option("-f", "--filter_data", dest="filter_data", default=None,
                  help="List of data to filter on")
parser.add_option("-o", "--operators", dest="operators", default=None,
                  help="List of operators for filtering")
parser.add_option("-t", "--thresholds", dest="thresholds", default=None,
                  help="List of thresholds for operators")
parser.add_option("-p", "--principal_components", dest="principal_components", default=None,
                  help="List of PCs to regress out")
parser.add_option("-r", "--regress_out", dest="regress_out", default=None,
                  help="Regress data on columns of matrix")
parser.add_option("--iterate_clean", dest="iterate_clean", default=False,
                  help="Repeat cleaning for 0 to n PCs specified by -p")
parser.add_option("--odir", dest="odirectory", default='./',
                  help="Output directory+prefix to prepend to any saved output")



(options, args) = parser.parse_args()

class Preprocessing:

    def __init__(self, data_source, transpose=config['transpose_data'], filter_data=None, operators=None, thresholds=None, clean_components=None, 
                    regress_out=None, sample_labels=None, feature_labels=None, iterate_clean=False, odir='./'):
        """
        UPDATE
        """

        self.data = self.load(data_source)
        self.s = self.load(sample_labels)
        self.f = self.load(feature_labels)

        # we can use feature labels and sample labels to fix the orientation of the matrix
        # the only special case is a square matrix, which then we can just make an assumption that features correspond to columns
        # otherwise we orient the matrix sample-rows, feature-cols
        # this will override transpose specified in config

        if transpose:
            self.data = self.data.T
        """
        if self.data.shape[0] != self.data.shape[1]:
            if self.s is not None:
                if self.data.shape[1] == self.s.size:
                    self.data = self.data.T
            elif self.f is not None:
                if self.data.shape[0] == self.f.size:
                    self.data = self.data.T
        """
        # if filter_data is strings, they are paths to pickles so load them
        # otherwise assume they are in some usable form (numpy array)
        if filter_data is not None and operators is not None and thresholds is not None:
            for i in range(len(filter_data)):
                if type(filter_data[i]) is str:
                    filter_data[i] = load(open(filter_data[i]))

            filters = zip(filter_data, operators, thresholds)
            for f in filters:
                self.data = self.filter(f)

        if iterate_clean:
            #cleans = np.empty(clean_components+1)
            for i in range(clean_components+1):
                c = np.arange(i)
                clean = self.clean(c, regress_out)
                if transpose:
                    clean = clean.T
                clean = scale(clean)
                pca = Representation(clean, 'pca').getRepresentation() 
                kmeans = Representation(clean, 'kmeans').getRepresentation()
                plt_title = "RemovedPCs:" + str(c)
                plot_2D(pca[0], kmeans[0], title=plt_title, savename=odir)                
                
        self.data = self.clean(clean_components, regress_out)

    def load(self, source):
        """
        Loads data from filename and returns it.
        """
        if type(source) is not str:
            # if what we are passed is not a string, assume it is the object we want it to be
            return source

        f, file_extension = os.path.splitext(source)
        
        if file_extension == '.p':
            # check if its a pickle
            data = load(open(source))
        else:
            # otherwise try and open it as text
            data = np.genfromtxt(source, comments='!', delimiter='\t')
            if np.any(np.isnan(data[0,:])):
                data = data[1:,:]
            if np.any(np.isnan(data[:,0])):
                data = data[:, 1:]
            #data = data[1::, 1::]  # Drop first row and column of label data since it just becomes NaN

        return data

    def filter(self, f, data=None, axis=None):

        operation = self.make_operation(f[1], f[2])
        include = operation(f[0])
        n = include.size

        if data == None:
            data = self.data
        if axis == None:
            # attempt to infer axis by dimension of filter output
            if data.shape[0] == data.shape[1]:
                # this is an ambigous case
                print >> sys.stderr, 'Ambiguous dimensions, please specify axis'
                return
            if data.shape[0] == n:
                axis = 0
            elif data.shape[1] == n:
                axis = 1
            else:
                print >> sys.stderr, 'Mismatched dimensions'
        if axis == 1:
            data = data.T 

        data = data[include, :]

        if axis == 1:
            data = data.T

        return data

    def make_operation(self, operator, threshold):
        """
        add operators we want to use here
        """
        if operator is '<':
            def f(x) : return x < threshold
        if operator is '<=':
            def f(x) : return x <= threshold
        if operator is '==':
            def f(x) : return x == threshold
        if operator is '>=':
            def f(x) : return x >= threshold
        if operator is '>':
            def f(x) : return x > threshold

        return f


    def pca_view(self):
        pca = Representation(self.view, 'pca').getRepresentation()
        plt.plot(pca[2])
        plt.show()
        pdb.set_trace()
        plt.clf()
        return pca

    def pca_view_diff(self):
        view = self.time_diff()
        pca = Representation(view, 'pca').getRepresentation()
        plt.plot(pca[2])
        plt.show()
        pdb.set_trace()
        plt.clf()
        return pca

    def time_diff(self):
        num_t = self.view.shape[1]
        num_g = self.view.shape[0]
        d_matrix = np.ndarray((num_g, num_t - 1))
        for i in range(1, num_t):
            d_matrix[:, i - 1] = self.view[:, i] - self.view[:, i - 1]
        return d_matrix

    def clean(self, components=None, regress_out=None):
        if components is None:
            components = config['clean_components']
        if type(components) is int:
            components = np.arange(components)
        
        print "Cleaning data..."
        data = scale(self.data)
        U, S, V = Representation(data, 'svd').getRepresentation()
        loadings = U[:, components]
        
        new_data = np.copy(data)
        # if there is additional data to regress against, do that first
        if regress_out is not None:
            new_data = self.regress_out(regress_out)
        # if we do actually have PCs to regress out do that now
        if loadings.size != 0:
            new_data = self.regress_out(loadings)
        #new_data = scale(new_data)

        return new_data

    def regress_out(self, X, y=None, axis=None):
        if y is None:
            y = self.data
        if axis is None:
            # try and figure out what axis based on the dimensions of independent variable
            n = X.shape[0]
            if y.shape[0] == y.shape[1]:
                # this is an ambigous case
                print >> sys.stderr, 'Ambiguous dimensions, please specify axis'
                return
            if y.shape[0] == n:
                axis = 0
            elif y.shape[1] == n:
                axis = 1
            else:
                print >> sys.stderr, 'Mismatched dimensions'
                
        print 'Axis: ', axis
        if axis == 1:
            y = np.transpose(y)
        linreg = LinearRegression()
        linreg.fit(X, y)
        residual = y - linreg.predict(X)

        if axis == 1:
              y = y.T
              residual = residual.T
        return residual

if __name__ == "__main__":

    files = []
    if options.dataset == 'my_connectome':
        files = ['GSE58122_varstab_data_prefiltered_geo.txt', 'mc_geneset.p', 'GSE58122_series_matrix.txt', 'rin.p']
        timesteps = np.arange(0,43)

    p = Preprocessing(options.directory+files[0], filter_data=options.filter_data, operators=options.operators, 
                        thresholds=options.thresholds, clean_components=options.principal_components, sample_labels=options.directory+files[1])
