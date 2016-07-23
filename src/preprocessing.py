import numpy as np
import pickle
from optparse import OptionParser
from sklearn.preprocessing import scale
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.linalg import svd as svd_func
# from tools.helpers import *


class Preprocessing:

    def __init__(self, data, sample_labels=None, feature_labels=None,
                 transpose=False, dtype=float, odir=''):
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
        self.raw = data.astype(dtype)  # keep a copy of the raw data for reset
        self.raw_samples = sample_labels
        self.raw_features = feature_labels

        self.data = pd.DataFrame(data=self.raw.copy(), index=self.raw_samples,
                                 columns=self.raw_features)

        if transpose:
            self.transpose()

        self.samples = self.data.axes[0].get_values()
        self.features = self.data.axes[1].get_values()
        # filter_data=None, operators=None, thresholds=None,
        # clean_components=None, regress_out=None,

    def filter(self, f, include_nan=True):
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

        # if data is None:
        #    data = self.data
        if include_nan:
            nan_mask = np.isnan(f_data)
            include = np.logical_or(include, nan_mask)

        included = np.empty(0)
        if f_axis == 1:
            self.data = self.data.iloc[:, include]
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

    def clean(self, components=None, regress_out=None, update_data=True,
              scale_in=True, scale_out=False):
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
            components = np.array([])
        if type(components) is int:
            components = np.arange(components)

        print "Cleaning data..."
        data = self.data.as_matrix()
        if scale_in:
            data = scale(data)
        U, S, V = svd_func(data, full_matrices=False)
        loadings = U[:, components]

        new_data = np.copy(data)
        # if there is additional data to regress the data against, do that now
        if regress_out is not None:
            for r in regress_out:
                new_data = self.regress_out(r[0], new_data, r[1])
        # if we do actually have PCs to regress out do that now
        if loadings.size != 0:
            # loadings = loadings.reshape(-1, len(components))
            new_data = self.regress_out(loadings, new_data, 0)

        if scale_out:
            new_data = scale(new_data)

        if update_data:
            self.data.iloc[:, :] = new_data

        return new_data

    def regress_out(self, X, y, axis):
        """
        regresses X on y
        by defauly y is the matrix in self.data
        if you don't specify an axis it assumes 0
        """
        print 'Regressing out variable...'

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
        self.data = pd.DataFrame(data=self.raw.copy(), index=self.raw_samples,
                                 columns=self.raw_features)
        self.samples = self.data.axes[0].get_values()
        self.features = self.data.axes[1].get_values()

    def scale(self, axis=0, with_mean=True, with_std=True):
        """
        Scale data so each feature has mean 0 variance 1
        """
        self.data.loc[:, :] = scale(self.data.as_matrix(), axis,
                                    with_mean=with_mean, with_std=with_std)

    def log_transform(self, smoothing):
        """
        log2 transform on data
        adds smoothing value to avoid log transform on non-positive values
        """
        self.data.loc[:, :] = np.log2(self.data.as_matrix() + smoothing)

    def transpose(self):
        self.data = self.data.T
        self.samples = self.data.index.values
        self.features = self.data.columns.values


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
    row_labels = None
    col_labels = None
    data = np.empty(0)

    if filetype == 'pickle':
        data = pickle.load(open(source))
    elif filetype == 'tsv':
        print source, ' as tab seperated'
        data = np.genfromtxt(source, comments='!', delimiter='\t',
                             dtype=object)
    elif filetype == 'csv':
        print source, ' as column seperated'
        data = np.genfromtxt(source, comments='!', delimiter=',',
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
    """
    Order of operations:
    1. load the file, create preprocessing object
    2. perform any filtering specified
    3. log transform data if specified, smoothin by value of log_smoothing
       (default 1)
    4. scale if specified (scaled columns to have mean 0 variance 1)
    5. data cleaning on pcs and values specified in regress_rows/regress_cols
    6. save output
    """

    parser = OptionParser()
    # input options
    parser.add_option("--in_directory", dest="in_dir", default="",
                      help="Directory to look for files, default curr directory")
    parser.add_option("-d", "--dataset", dest="dataset", default=None,
                      help="Dataset, the file the the data matrix in it")
    parser.add_option("--filetype", dest="filetype", default='pickle',
                      help="File type of data file ('csv', 'tsv', 'pickle')")

    parser.add_option("--has_row_labels", action="store_true",
                      dest="has_row_labels", default=False,
                      help="If True, first column of dataset file are labels")
    parser.add_option("--has_col_labels", action="store_true",
                      dest="has_col_labels", default=False,
                      help="If True, first column of dataset file are labels")

    # transpose option
    parser.add_option("--transpose_data",  action="store_true",
                      dest="transpose", default=False,
                      help="If True, tranpose the data matrix")

    # log-transform options
    parser.add_option("-l", "--log_transform", action="store_true",
                      dest="log_transform", default=False,
                      help="If true performs log2 transform on data")
    parser.add_option("--smoothing", default=1, dest="smoothing",
                      help="Smoothing value for log trasform")

    # scaling options
    parser.add_option("-s", "--scale_data", action="store_true",
                      dest="scale_data", default=False,
                      help="If true scales data (default to mean 0 variance 1)")
    parser.add_option("--scale_axis", dest="scale_axis", default=0,
                      help="Axis to scale one")
    parser.add_option("--center_off", action="store_false",
                      dest="center_off", default=True,
                      help="If enabled scale won't attempt to center data")
    parser.add_option("--unit_std_off", action="store_false",
                      dest="unit_std_off", default=True,
                      help="If enabled scale won't scale to unit-normal")

    # filtering options
    parser.add_option("-f", "--filter_data", dest="filter_data", default=None,
                      help="List of data to filter on")
    parser.add_option("-o", "--operators", dest="operators", default=None,
                      help="List of operators for filtering")
    parser.add_option("-t", "--thresholds", dest="thresholds", default=None,
                      help="List of thresholds for operators")
    parser.add_option("-a", "--filter_axes", dest="filter_axes", default=None,
                      help="List of axes for each filter operation")

    # cleaning options
    parser.add_option("-p", "--principal_components", dest="principal_components",
                      default=[], help="List of PCs to regress out")
    parser.add_option("--regress_cols", dest="regress_cols", default=None,
                      help="Regress data on columns of matrix")
    parser.add_option("--regress_rows", dest="regress_rows", default=None,
                      help="Regress data on rows of matrix")

    # output options
    parser.add_option("--odir", dest="out_directory", default='',
                      help="Output directory+prefix to prepend to any saved output"
                      )
    parser.add_option("--saveas", dest="saveas", default='pickle',
                      help=("How to save data:" +
                            "\'pickle\', \'mat\', \'R\', \'txt\'")
                      )

    (options, args) = parser.parse_args()

    datapath = options.in_dir + options.dataset

    # has_row_labels = False
    # has_row_labels = False

    # if options.has_row_labels == 'True':
    #    has_row_labels = True

    # if options.has_col_labels == 'True':
    #    has_col_labels = True

    data, row_labels, col_labels = load_file(datapath, options.filetype,
                                             options.has_row_labels,
                                             options.has_col_labels)

    print row_labels
    print col_labels
    p = Preprocessing(data, row_labels, col_labels, options.transpose, float,
                      options.out_directory)

    filter_data = options.filter_data
    operators = options.operators
    thresholds = options.thresholds
    filter_axes = options.filter_axes

    if filter_data is not None:
        for i in range(len(filter_data)):
            filter_data[i] = pickle.load(open(filter_data[i]))

        filters = zip(filter_data, operators, thresholds, filter_axes)
        for f in filters:
            p.filter(f)

    if options.log_transform:
        p.log_transform(options.smoothing)

    if options.scale_data:
        p.scale(options.scale_axis, options.center_off, options.unit_std_off)
    regress_rows = options.regress_rows
    regress_cols = options.regress_cols
    pc = options.principal_components

    regress_out = []
    if regress_rows is not None:
        regress_out.append([(pickle.load(open(rr)), 1) for rr in regress_rows])
    if regress_cols is not None:
        regress_out.append([(pickle.load(open(rc)), 0) for rc in regress_cols])
    p.clean(pc, regress_out)

    savename = options.out_directory + 'preprocessed_data'
    print "Saving processed data to: ", savename

    if options.saveas is 'pickle':
        pickle.dump(p.data, open(savename, 'wb'))
    if options.saveas is 'csv':
        p.data.to_csv(path_or_buf=savename, sep=',')
    if options.saveas is 'tsv':
        p.data.to_csv(path_or_buf=savename, sep='\t')

    # TODO: be able to save for use in r and matlab
    # if options.saveas is 'r':
    #    from rpy2.robjects import pandas2ri
    #    pandas2ri.activate()
    #    r_df = padas2ri.pi2ri(p.data)
    # if options.saveas is 'mat':
    #    import scipy.io as sio
