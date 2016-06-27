from representation import Representation
# from GP import fit_gp_with_priors
import numpy as np
import sys
from pickle import load, dump
from config import config, param, get_org_params
from optparse import OptionParser
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pdb
from gsea.run_enrichment import run_enrichment
from gsea.process_gsea_output import get_sigs
from tools.helpers import make_config_string
import os

parser = OptionParser()
parser.add_option("-d", "--dataset", dest="dataset", default=None,
                  help="Dataset")
parser.add_option("--dir", dest="directory",
                  help="Directory to look for data files.")

(options, args) = parser.parse_args()


class ClusterPipeline:

    def __init__(self, rep_type, view1, view2, genes):
        self.rep_type = rep_type
        self.view1 = view1
        self.view2 = view2
        self.genes = genes

    def run_pipeline(self):
        if self.rep_type == 'pca':
            view1_loadings = self.learn_loadings(self.view1)
            # view2_loadings = self.learn_loadings(self.view2, 'pca')
        elif self.rep_type == 'cca':
            if self.view1.shape[1] == self.view2.shape[1] + 1:
                loadings = self.learn_loadings(self.view1[:, 1:], view2=self.view2)
            else:
                loadings = self.learn_loadings(self.view1, view2=self.view2)
            view1_loadings = loadings[0]
            # view2_loadings = loadings[1]
        clusters, centers = Representation(view1_loadings, 'kmeans', scale=False).getRepresentation()
        cluster_dict = self.process_clusters(clusters, centers)
        return cluster_dict

    def learn_loadings(self, view, view2=None):
        if self.rep_type == 'pca':
            loadings = self.learn_pca_loadings(view)
        elif self.rep_type == 'cca':
            loadings = self.learn_cca_loadings(view, view2)
        return loadings

    def learn_pca_loadings(self, view):
        view_dim1 = view.shape[1]
        svd = Representation(view, 'svd', axis=0)
        svd_results = svd.getRepresentation()
        U = svd_results[0]
        S_vector = svd_results[1]
        # V = svd _results[2]
        S = np.zeros((view_dim1, view_dim1), dtype=float)
        S[:view_dim1, :view_dim1] = np.diag(S_vector)
        svd_loadings = np.dot(U, S)
        return svd_loadings[:, :param['components']]

    def learn_cca_loadings(self, view1, view2):
        cca = Representation(view1, 'cca', axis=0, data2=view2)
        cca_result = cca.getRepresentation()
        cca_loadings = cca_result
        return cca_loadings

    def process_clusters(self, clusters, cluster_centers):
        cluster_dict = {}
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
        self.dir = config['project_root'] + 'data/trajectories/'
        self.dir += make_config_string(param, get_org_params())
        try:
            os.mkdir(self.dir)
        except:
            pass

    def run_pipeline(self, clusters, sig_bps):
        cluster_time_series = self.generate_time_series(clusters, sig_bps)
        for cluster_name, bp_dict in cluster_time_series.items():
            for bp_name, trajectories in bp_dict.items():
                bp_name = bp_name.strip()
                print cluster_name, bp_name
                print "Number of genes:", len(trajectories)
                self.plot_avg_trajectory(timesteps, trajectories)
                plt.savefig(self.dir + '/' + cluster_name + '_' + bp_name + '_avg.png')
                self.plot_all_trajectories(timesteps, trajectories)
                plt.savefig(self.dir + '/' + cluster_name + '_' + bp_name + '_all.png')
                #fit_gp_with_priors(trajectory[0], self.timesteps)

    def plot_avg_trajectory(self, timesteps, trajectories):
        avg_trajectory = (sum(trajectories)) / float(len(trajectories))
        plt.clf()
        plt.plot(timesteps, avg_trajectory)
        # plt.show()
        # pdb.set_trace()

    def plot_all_trajectories(self, timesteps, trajectories):
        plt.clf()
        for trajectory in trajectories:
            plt.plot(timesteps, trajectory)
        # plt.show()
        # pdb.set_trace()

    def generate_time_series(self, cluster_dict, sig_bp_dict):
        new_cluster_dict = {}
        for cluster_name, cluster_genes in cluster_dict.items():
            gsea_filename = config['gsea_file_prefix'] + cluster_name
            if sig_bp_dict.get(gsea_filename) is not None:

                cluster_time_series = {}
                vectors = []
                for gene in cluster_genes:
                    index = self.genes.index(gene)
                    vectors.append(self.time_series[index, :])
                cluster_time_series['TOTAL'] = vectors  # TODO: scale?

                for bp in sig_bp_dict.get(config['gsea_file_prefix'] + cluster_name):
                    bp_genes = bp[-1].split(',')
                    bp_name = bp[0]
                    bp_vectors = []
                    for gene in bp_genes:
                        index = self.genes.index(gene)
                        bp_vectors.append(self.time_series[index, :])
                    cluster_time_series[bp_name] = bp_vectors  # TODO: scale?

                new_cluster_dict[cluster_name] = cluster_time_series

        return new_cluster_dict


class BasicPipeline:

    def __init__(self, view, genes):
        self.view = view
        self.genes = genes

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

    def clean(self):
        print "Cleaning data..."
        view = scale(self.view)
        U, S, V = Representation(view, 'svd').getRepresentation()
        new_view = np.ndarray(view.shape)
        loadings = U[:, 0:param['clean_components']]
        for i in range(view.shape[1]):
            feature_vector = view[:, i]
            model = LinearRegression(fit_intercept=False)
            model.fit(loadings, feature_vector)
            residual = feature_vector - model.predict(loadings)
            new_view[:, i] = residual

        new_view = scale(new_view)

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

    param['dataset'] = options.dataset
    identifying_string = make_config_string(param, get_org_params())

    raw_view1 = np.transpose(load(open(options.directory + files[0])))
    raw_view2 = np.transpose(load(open(options.directory + files[1])))
    genes = load(open(options.directory + files[2]))

    # histogram_analysis(raw_view1, 'view1')
    # histogram_analysis(raw_view2, 'view2')
    # sys.exit()

    if param['log_transform']:
        basic_view1 = BasicPipeline(np.log2(raw_view1 + 2), genes)
        basic_view2 = BasicPipeline(np.log2(raw_view2 + 2), genes)
    else:
        basic_view1 = BasicPipeline(raw_view1, genes)
        basic_view2 = BasicPipeline(raw_view2, genes)

    if config['view_pca_plots']:
        print "Displaying PCA results on uncleaned data..."
        if not param['time_transform']:
            basic_view1.pca_view()
            basic_view2.pca_view()
        else:
            basic_view1.pca_view_diff()
            basic_view2.pca_view_diff()

    print "Cleaning data..."
    if param['clean_data']:
        view1 = basic_view1.clean()
        view2 = basic_view2.clean()
        basic_view1.view = view1
        basic_view2.view = view2

    if config['view_pca_plots']:
        print "Displaying PCA results on cleaned data..."
        if not config['time_transform']:
            basic_view1.pca_view()
            basic_view2.pca_view()
        else:
            basic_view1.pca_view_diff()
            basic_view2.pca_view_diff()

    if config['run_clustering']:
        print "Running cluster pipeline..."
        cluster_pipeline = ClusterPipeline(param['representation'], view1, view2, genes)
        clusters = cluster_pipeline.run_pipeline()

        print "Dumping clusters for enrichment analysis..."
        dump(clusters, open('dumps/' + identifying_string + '.dump', 'w'))
    else:
        print "Loading clusters for enrichment analysis..."
        clusters = load(open('dumps/' + identifying_string + '.dump'))

    if config['run_enrichment']:
        print "Running gene set enrichment analyses..."
        run_enrichment(clusters, genes, config['project_root'] + 'src/gsea/GO-BP.gmt', config['project_root'] + 'data/gsea_output/' + identifying_string + '/')

    print "Processing GSEA output into significant enrichments..."
    output_dir = config['project_root'] + 'data/gsea_output/' + identifying_string + '/'
    sig_bps = get_sigs(output_dir, param['gsea_p_value_thresh'], param['gsea_fdr_thresh'])

    print "Running GP pipeline..."
    if config['trajectories'] == 'cleaned':
        gp_pipeline = GPPipeline(view1, genes, timesteps)
    elif config['trajectories'] == 'raw':
        gp_pipeline = GPPipeline(raw_view1, genes, timesteps)
    gp_pipeline.run_pipeline(clusters, sig_bps)
