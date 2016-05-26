from representation import Representation
from GP import fit_gp_with_priors
import numpy as np
import sys
from pickle import load, dump
from config import config
from optparse import OptionParser
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.io import savemat


parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset", default=None,
                  help="Dataset")
parser.add_option("--dir", dest="directory",
                  help="Directory to look for data files.")

(options, args) = parser.parse_args()


class ClusterPipeline:

    def __init__(self, view1, view2, genes):
        self.view1 = view1
        self.view2 = view2
        self.genes = genes

    def run_pipeline(self):
        if config['time_transform']:
            self.view1 = self.preprocess(self.view1)
            self.view2 = self.preprocess(self.view2)
        view1_pca_loadings = self.learn_loadings(self.view1, 'pca')
        view2_pca_loadings = self.learn_loadings(self.view2, 'pca')
        if self.view1.shape[1] == self.view2.shape[1] + 1:
            cca_loadings = self.learn_loadings(self.view1[:, 1:], 'cca', view2=self.view2)
        else:
            cca_loadings = self.learn_loadings(self.view1, 'cca', view2=self.view2)
        view1_cca_loadings = cca_loadings[0]
        view2_cca_loadings = cca_loadings[1]
        pca_clusters, pca_centers = Representation(view1_pca_loadings, 'kmeans', scale=False).getRepresentation()
        cca_clusters, cca_centers = Representation(view1_cca_loadings, 'kmeans', scale=False).getRepresentation()
        pca_cluster_dict = self.process_clusters(pca_clusters, pca_centers)
        cca_cluster_dict = self.process_clusters(cca_clusters, cca_centers)
        return pca_cluster_dict, cca_cluster_dict

    def preprocess(self, view):
        num_t = view.shape[1]
        num_g = view.shape[0]
        d_matrix = np.ndarray((num_g, num_t - 1))
        for i in range(1, num_t):
            d_matrix[:, i - 1] = view[:, i] - view[:, i - 1]
        return d_matrix

    def learn_loadings(self, view, rep_type, view2=None):
        if rep_type == 'pca':
            loadings = self.learn_pca_loadings(view)
        elif rep_type == 'cca':
            loadings = self.learn_cca_loadings(view, view2)
        return loadings

    def learn_pca_loadings(self, view):
        view_dim1 = view.shape[1]
        svd = Representation(view, 'svd', axis=0)
        svd_results = svd.getRepresentation()
        U = svd_results[0]
        S_vector = svd_results[1]
        # V = svd_results[2]
        S = np.zeros((view_dim1, view_dim1), dtype=float)
        S[:view_dim1, :view_dim1] = np.diag(S_vector)
        svd_loadings = np.dot(U, S)
        # import pdb; pdb.set_trace()
        return svd_loadings[:, :config['pca_components']]

    def learn_cca_loadings(self, view1, view2):
        cca = Representation(view1, 'cca', axis=0, data2=view2)
        cca_result = cca.getRepresentation()
        cca_loadings = cca_result
        return cca_loadings

    def process_clusters(self, clusters, cluster_centers):
        cluster_dict = {}
        import pdb; pdb.set_trace()
        for i in range(len(clusters)):
            cluster_label = "c" + str(clusters[i])
            gene = self.genes[i]
            if cluster_dict.get(cluster_label) is None:
                cluster_dict[cluster_label] = []
            cluster_dict[cluster_label].append(gene)
        for key, value in cluster_dict.items():
            cluster_dict[key] = np.asarray(value, dtype=np.object)
        return cluster_dict


class GPPipeline:

    def __init__(self, time_series, genes, timesteps):
        self.time_series = scale(time_series)
        self.genes = genes
        self.num_timesteps = len(timesteps)
        self.timesteps = timesteps

    def run_pipeline(self, clusters):
        cluster_time_series = self.generate_time_series(clusters)
        pca_sig_clusters = ['c11', 'c61']
        cca_sig_clusters = ['c12', 'c40', 'c44', ]
        for cluster_name, trajectory in cluster_time_series.items():
            if cluster_name in cca_sig_clusters:
                print cluster_name
                print "Number of genes:", len(clusters[cluster_name])
                plt.plot(timesteps, trajectory[0])
                plt.show()
                import pdb; pdb.set_trace()
                #fit_gp_with_priors(trajectory[0], self.timesteps)

    def generate_time_series(self, cluster_dict):
        cluster_time_series = {}
        for cluster_name, cluster_genes in cluster_dict.items():
            sum_vector = np.zeros((1, self.num_timesteps))
            for gene in cluster_genes:
                index = self.genes.index(gene)
                sum_vector += self.time_series[index, :]
            cluster_time_series[cluster_name] = sum_vector / len(cluster_genes)
        return cluster_time_series


class BasicPipeline:

    def __init__(self, view, genes):
        self.view = view
        self.genes = genes

    def pca_view(self):
        pca = Representation(self.view, 'pca').getRepresentation()
        plt.plot(pca[2])
        plt.show()
        import pdb; pdb.set_trace()
        plt.clf()
        return pca

    def pca_view_diff(self):
        view = self.time_diff()
        pca = Representation(view, 'pca').getRepresentation()
        plt.plot(pca[2])
        plt.show()
        import pdb; pdb.set_trace()
        plt.clf()
        return pca

    def time_diff(self):
        num_t = self.view.shape[1]
        num_g = self.view.shape[0]
        d_matrix = np.ndarray((num_g, num_t - 1))
        for i in range(1, num_t):
            d_matrix[:, i - 1] = self.view[:, i] - self.view[:, i - 1]
        return d_matrix

    def clean(self):
        print "Cleaning data..."
        view = scale(self.view)
        U, S, V = Representation(view, 'svd').getRepresentation()
        new_view = np.ndarray(self.view.shape)
        loadings = U[:, 0:config['clean_components']]
        for i in range(view.shape[1]):
            feature_vector = view[:, i]
            model = LinearRegression(fit_intercept=True)
            model.fit(loadings, feature_vector)
            residual = feature_vector - model.predict(loadings)
            new_view[:, i] = residual

        return new_view


if __name__ == "__main__":
    if options.dataset == 'hRSV':
        files = ['h_sapiens_matrix.dump', 'hRSV_matrix.dump', 'h_sapiens_genes.dump']
        timesteps = [0, 2, 4, 8, 12, 16, 20, 24]
    elif options.dataset == 'myeloma':
        files = ['polyA.dump', 'ribosome.dump', 'myeloma_genes.dump']
        timesteps = [0, 1.5, 3, 6, 9, 12]
    elif options.dataset is None:
        sys.exit('Please supply a dataset to use.')

    raw_view1 = np.transpose(load(open(options.directory + files[0])))
    raw_view2 = np.transpose(load(open(options.directory + files[1])))
    genes = load(open(options.directory + files[2]))

    basic_view1 = BasicPipeline(np.log2(raw_view1 + 2), genes)
    basic_view2 = BasicPipeline(np.log2(raw_view2 + 2), genes)
    print "PCA results on uncleaned data:"
    """
    basic_view1.pca_view()
    basic_view1.pca_view_diff()
    basic_view2.pca_view()
    basic_view2.pca_view_diff()
    """
    print "Cleaning data..."
    view1 = basic_view1.clean()
    view2 = basic_view2.clean()

    print "PCA results on cleaned data:"

    basic_view1.view = view1
    basic_view2.view = view2
    # basic_view1.pca_view()
    # basic_view1.pca_view_diff()
    # basic_view2.pca_view()
    # basic_view2.pca_view_diff()

    print "Running cluster pipeline..."

    cluster_pipeline = ClusterPipeline(view1, view2, genes)
    pca_clusters, cca_clusters = cluster_pipeline.run_pipeline()

    print "Dumping clusters for enrichment analysis..."
    dump(pca_clusters, open(options.dataset + '_pca_clusters_cleaned=' + config['clean_components'] + ',components=' + config['pca_components'] + ',cca_reg=' + config['cca_reg'] + '.dump', 'w'))
    dump(cca_clusters, open(options.dataset + '_cca_clusters_cleaned=' + config['clean_components'] + ',components=' + config['cca_components'] + ',cca_reg=' + config['cca_reg'] + '.dump', 'w'))

    savemat(open('../data/' + options.dataset + '_pca_clusters_cleaned=' + config['clean_components'] + ',components=' + config['pca_components'] + ',cca_reg=' + config['cca_reg'] + '.mat', 'w'), pca_clusters)
    savemat(open('../data/' + options.dataset + '_cca_clusters_cleaned=' + config['clean_components'] + ',components=' + config['cca_components'] + ',cca_reg=' + config['cca_reg'] + '.mat', 'w'), cca_clusters)
    savemat(open('../data/' + options.dataset + '_genes.mat', 'w'), {'genes': genes})
    """
    pca_clusters = load(open(options.dataset + '_pca_clusters_cleaned=' + config['clean_components'] + ',components=' + config['pca_components'] + ',cca_reg=' + config['cca_reg'] + '.dump'))
    cca_clusters = load(open(options.dataset + '_cca_clusters_cleaned=' + config['clean_components'] + ',components=' + config['cca_components'] + ',cca_reg=' + config['cca_reg'] + '.dump'))

    print "Running GP pipeline..."
    gp_pipeline = GPPipeline(view1, genes, timesteps)
    gp_pipeline.run_pipeline(cca_clusters)
    """