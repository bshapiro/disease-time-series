import numpy as np
import sys
import pickle
from config import config, param
from optparse import OptionParser
from sklearn.preprocessing import scale
import pandas as pd
from sklearn.linear_model import LinearRegression
from tools.helpers import *
from representation import Representation

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
parser.add_option("-a", "--filter_axes", dest="filter_axes", default=None,
                  help="List of axes for each filter operation")
parser.add_option("-p", "--principal_components", dest="principal_components",
                  default=None, help="List of PCs to regress out")
parser.add_option("--regress_cols", dest="regress_cols", default=None,
                  help="Regress data on columns of matrix")
parser.add_option("--regress_rows", dest="regress_rows", default=None,
                  help="Regress data on columns of matrix")
parser.add_option("--odir", dest="odirectory", default='./',
                  help="Output directory+prefix to prepend to any saved output"
                  )
parser.add_option("--saveas", dest="saveas", default='pickle',
                  help="How to save preprocessed data: \'pickle\', \'mat\', \'R\', \'txt\'"
                  )

(options, args) = parser.parse_args()


class Preprocessing:

    def __init__(self, data, sample_labels=None, feature_labels=None, transpose=config['transpose_data'], iterate_clean=False, odir='./'):
        """
        UPDATE
        Notes:
        - the only arguments that the main object should take are data-specific
        (not pipeline-specific)
        e.g. data configuration, samples/features, matrix itself, transpose,
        etc.
        - arguments that relate to actual cleaning steps should be limited to
        function arguments for malleable
        pipeline use, e.g. thresholds, cleaned components, etc.
        """
        # self.data = self.load(data_source)
        self.raw = data  # keep a copy of the raw data for reset
        self.raw_samples = sample_labels
        self.raw_features = feature_labels

        if transpose:
            self.raw = self.raw.T

        self.data = pd.DataFrame(data=self.raw, index=self.raw_samples,
                                 columns=self.raw_features)
        self.samples = self.data.axes[0].get_values()
        self.features = self.data.axes[1].get_values()
        # filter_data=None, operators=None, thresholds=None,
        # clean_components=None, regress_out=None,

    def filter(self, f, include_nan=config['filter_include_nan']):
        """
        f contains all the information necessary for filtering
        - f[0] is the data to filter on
        - f[1] is a valid filtering operation
        - f[2] is a threshold for the filtering operation
        - f[3] axis to filtering on 0(rows/samples) or 1(cols/feautres)

        include_nan, specified in config_include_nan says whether to filter out
        nan values. nan values may appeaer because the filter_data must
        correspond with axes of the data matrix and you may not always have
        data on the condition you want to filter on for all samples/features

        the method updates self.data DataFrame to include only rows/columns
        that satisfy the filter
        returns an array of the row/column elements included by the filter
        """
        # TODO: this works well for filtering on conditions in the data
        # or filtering on data outside the data once
        # but if we want to filter on the same axis multiple times
        # we need a way to keep track of removed indices since the
        # user wont be able to easily recover which indices of the
        # external data to include in the filtering process
        # a solution to this is to add all the values they want to filter
        # on to the self.data DataFrame to perform filtering then remove them    
        f_data = f[0]
        f_operation = f[1]
        f_threshold = f[2]
        f_axis = f[3]

        operation = self.make_operation(f_operation, f_threshold)
        include = operation(f_data)

        #if data is None:
        #    data = self.data
        if include_nan:
            nan_mask = np.isnan(f_data)
            include = np.logical_or(include, nan_mask)

        included = np.empty(0)
        if f_axis == 1:
            self.data = self.data.iloc[:,include]
            included = self.data.columns.get_values()
        if f_axis == 0:
            self.data = self.data.iloc[include, :]
            included = self.data.index.get_values()

        self.samples = self.data.axes[0].get_values()
        self.features = self.data.axes[1].get_values()
        return included

    def make_operation(self, operator, threshold):
        """
        add operators we want to use here
        """
        if operator is '<':
            return lambda x: x < threshold
        if operator is '<=':
            return lambda x: x <= threshold
        if operator is '==':
            return lambda x: x == threshold
        if operator is '>=':
            return lambda x: x >= threshold
        if operator is '>':
            return lambda x: x > threshold

    def clean(self, components=None, regress_out=None, update_data=True):
        """
        - components is an int/list of PCs to regress out from the data
          if you give an int n it will regress out the top n components
          Note: default components to remove can be specified in config's
          'clean_components'
        - regress_out is a list of additional vectors to regress out of data
          format regress out as such [(array, axis), .. (array, axis)]
          so that for each regressor you specify the axis wrt the data
        - clean will compute the components, regress out the spcefied
          regressors and then regress out the PCs jointly
        - update_data, if true replaces the data matrix in self.data with
          cleaned data (default is true)
        returns matrix of cleaned data
        """
        if components is None:
            components = param['clean_components']
        if type(components) is int:
            components = np.arange(components)

        print "Cleaning data..."
        data = scale(self.data.as_matrix())
        U, S, V = Representation(data, 'svd').getRepresentation()
        loadings = U[:, components]

        new_data = np.copy(data)
        # if there is additional data to regress the data against, do that now
        if regress_out is not None:
            for r in regress_out:
                new_data = self.regress_out(r[0], new_data, r[1])
        # if we do actually have PCs to regress out do that now
        if loadings.size != 0:
            new_data = self.regress_out(loadings)

        new_data = scale(new_data)
        if update_data:
            self.data.iloc[:, :] = new_data

        return new_data

    def regress_out(self, X, y=None, axis=0):
        """
        regresses X on y
        by defauly y is the matrix in self.data
        if you don't specify an axis it assumes 0
        """
        print 'Regressing out variable...'
        if y is None:
            y = self.data

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

    def reset(self):
        """
        resets self.data to the original, raw data matrix
        """
        self.data = pd.DataFrame(data=self.raw, index=self.raw_samples,
                                 columns=self.raw_features)
        self.samples = self.data.axes[0].get_values()
        self.features = self.data.axes[1].get_values()


def load_file(source, filetype, has_row_labels=False, has_col_labels=False):
    """
    Loads data from filename and returns it.
    Must specifiy filetype 'csv', 'tsv', 'pickle'
    We assum txt files are comma or tab delimited
    if has_row_labels is true it takes the first column to be labels
    if has_col_labels is true it takes the first row to be labels
    format data to fit these specs
    Labels will be empty arrays if row_labels and col_labels are False
    Returns data matrix, row_labels, and column labels
    """
    row_labels = np.empty(0)
    col_labels = np.empty(0)
    data = np.empty(0)

    if filetype == 'pickle':
        data = pickle.load(open(source))
    else:
        data = np.genfromtxt(source, comments='!', delimiter=filetype,
                             dtype=object)
    # else:
    #    print >> sys.stderr, 'specify a valid filtype:',
    #    ' \'pickle\', \'csv\', \'tsv\''

    if has_row_labels and has_col_labels:
        row_labels = data[1:, 0]
        col_labels = data[0, 1:]
        data = data[1:, 1:]

    elif has_row_labels:
        row_labels = data[:, 0]
        data = data[:, 1:]

    elif has_col_labels:
        col_labels = data[0, :]
        data = data[1:, :]

    return data, row_labels, col_labels


if __name__ == "__main__":
    pass
    """
    # TODO: walk through pipeline using provided arguments
    # TODO: rewrite the data to some specified location (additional argument)

    files = []
    if options.dataset == 'my_connectome':
        files = ['GSE58122_varstab_data_prefiltered_geo.txt',
                 'mc_geneset.p', 'GSE58122_series_matrix.txt', 'rin.p']
        timesteps = np.arange(0, 43)

    p = Preprocessing(options.directory+files[0], filter_data=options.filter_data, operators=options.operators,
                      thresholds=options.thresholds, clean_components=options.principal_components, sample_labels=options.directory+files[1])

        # TODO: take out
        # if filter_data is strings, they are paths to pickles so load them
        # otherwise assume they are in some usable form (numpy array)
        if filter_data is not None and operators is not None and thresholds is not None:
            for i in range(len(filter_data)):
                if type(filter_data[i]) is str:
                    filter_data[i] = load(open(filter_data[i]))

            filters = zip(filter_data, operators, thresholds)
            for f in filters:
                self.data = self.filter(f)


        self.data = self.clean(clean_components, regress_out)
    """
