# import addpath
from collections import defaultdict
from sklearn.cluster import KMeans
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy as sp
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable


def export_clusters(cluster_labels, gene_set,
                    savename='../out/clustered_genes.p'):
    num_clusters = np.bincount(cluster_labels).size
    cluster_dict = {}
    for i in range(num_clusters):
        cluster_label = 'c' + str(i)
        if cluster_dict.get(cluster_label) is None:
            genes = gene_set[np.where(cluster_labels == i)[0]]
            cluster_dict[cluster_label] = genes

    pickle.dump(cluster_dict, open(savename, 'w'))
    # sio.savemat(savename, cluster_dict)


def plot_2D(data, color_labels=None, x=0, y=1, xlabel='X', ylabel='Y',
            title='2-Dimensional Plot', savename='plot'):
    fig = plt.figure()
    x_vals = data[:, x]
    y_vals = data[:, y]
    colors = label_coloring(color_labels)
    if colors is not None:
        plt.scatter(x_vals, y_vals, figure=fig, s=50, c=colors)
    else:
        plt.scatter(x_vals, y_vals, figure=fig, s=50)

    plt.title(title, figure=fig)
    plt.xlabel(xlabel, figure=fig)
    plt.ylabel(ylabel, figure=fig)
    savename = savename + title
    plt.savefig(savename)
    plt.close()
    return


def PCPlot_kmeans(data, pc1=0, pc2=1, k=3, xlabel='PC-1', ylabel='PC-2',
                  title='PC-Plot', odir='plot'):
    """
    Plot over PCs (1st and 2nd) coloring based on kmeans clustering
    """
    pca = PCA()
    pca.fit(data)
    pca_data = pca.transform(data)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    fig = plt.figure()
    plt.title(title, figure=fig)

    ax1 = fig.add_subplot(211)
    x_vals = pca_data[:, pc1]
    y_vals = pca_data[:, pc2]
    colors = label_coloring(kmeans.labels_)
    ax1.scatter(x_vals, y_vals, s=50, c=colors)
    ax1.scatter(x_vals, y_vals, s=50)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2 = fig.add_subplot(212)
    ax2.plot(pca[2])
    ax2.set_xlabel('PC')
    ax2.set_ylabel('Explained Variance Ratio')

    savename = odir + title
    plt.savefig(savename)
    plt.close()
    return


def PCPlot(data, pc1=0, pc2=1, labels=None, xlabel='PC-1', ylabel='PC-2',
           title='PC-Plot', odir='./'):
    """
    Plot over PCs (1st and 2nd) coloring based on kmeans clustering
    """
    pca = PCA()
    pca.fit(data)
    pca_data = pca.transform(data)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)

    x_vals = pca_data[:, pc1]
    y_vals = pca_data[:, pc2]
    pcplot = ax[0].scatter(x_vals, y_vals, s=50, c=labels)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)

    div = make_axes_locatable(ax[0])
    cax = div.append_axes("right", size="15%", pad=0.05)
    cbar = plt.colorbar(pcplot, cax=cax, ticks=np.arange(0, 10, .1),
                        format="%.2g")

    ax[1].plot(pca.explained_variance_ratio_)
    ax[1].set_xlabel('PC')
    ax[1].set_ylabel('Explained Variance Ratio')

    savename = odir + title
    plt.tight_layout()
    # plt.show()
    plt.savefig(savename)
    plt.close()
    return


def HeatMap(data, row_labels, col_labels, cmin=None, cmax=None,
            title='Heat Map', odir='./', x_rot='vertical'):
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(data)
    plt.clim(cmin, cmax)
    fig.suptitle(title)
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(col_labels, rotation=x_rot, minor=False)
    ax.set_yticklabels(row_labels, minor=False)
    plt.colorbar()
    plt.tight_layout()

    # plt.show()
    plt.savefig(odir+title)
    plt.close()


def QQPlot(observed_p, plt_title, savepath):
    # make the QQ plot, borrowed from Princy's code demonstrating QQ plots
    n = observed_p.size
    obs_p = -np.log10(np.sort(observed_p))
    th_p = np.arange(1/float(n), 1+(1/float(n)), 1/float(n))
    th_p = -np.log10(th_p)
    fig, ax = plt.subplots(ncols=1)
    ax.scatter(th_p, obs_p)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)
    ax.set_xlabel('Theoretical')
    ax.set_ylabel('Observed')
    ax.set_title(plt_title)
    savepath = savepath + plt_title
    plt.savefig(savepath)
    plt.close()


def benjamini(x, p):
    x = np.sort(x.flatten())
    p = p / x.size
    significant = np.empty(x.size, dtype=bool)
    for i in range(x.size):
        significant[i] = x[i] < (i+1) * p
    return significant


def label_coloring(labels):
    """
    returns a coloring where each unique label gets its own color
    """
    if labels is None:
        return None
    num_colors = np.bincount(labels).size
    color_vals = np.linspace(0, 1, num_colors, endpoint=True)
    color = np.empty(labels.size)
    for i in range(0, num_colors):
        color[labels == i] = color_vals[i]
    return color


def get_lsv(X, pca):
    """
    make sure mean of a feature across samples is 0
    """
    n = pca.explained_variance_.size

    # matrix of singular values
    SV = np.zeros((n, n))
    SV[np.diag_indices(n)] = pca.explained_variance_
    SV = np.sqrt(SV)

    # X = U SV W.T, where the columns of W are are eigenvectors/PCs
    # since rows of pca.components_ are PCs pca.components_ = W.T
    W = pca.components_.T
    # U = X W SV^-1, since W W.T = I
    U = X.dot(W).dot(np.linalg.inv(SV))
    return U, SV, W


def associate(data1, data2, targets1=None, targets2=None, method='spearman',
              save=False, savename='',  outpath=''):
    """
    data1 (ixj)and data2(nxm) are DataFrames
    targets are the columns of data 2 that we want to check an
    associationg for with the rows of data1
    returns an ixm dataframe with benjamini hochberg corrected p values of
    spearman correlations
    """
    print 'Finding Associations...'
    if targets1 is None:
        targets1 = data1.axes[0]
    if targets2 is None:
        targets2 = data2.axes[1]

    pvals = pd.DataFrame(data=np.empty((targets1.size, targets2.size)),
                         index=targets1, columns=targets2)
    corr = pd.DataFrame(data=np.empty((targets1.size, targets2.size)),
                        index=targets1, columns=targets2)

    if method is 'spearman':
        method = sp.stats.spearmanr
    if method is 'pearson':
        method = sp.stats.pearsonr

    for j in targets2:
        for i in targets1:
            notnan = data2[(~np.isnan(data2[j].astype(float)))].index
            r, p = method(data1.loc[i, notnan].as_matrix(),
                          data2.loc[notnan, j].as_matrix())
            pvals.loc[i, j] = p
            corr.loc[i, j] = r
        plttitle = savename + j
        savepath = outpath + savename + j
        print "saving plot to: ", savepath
        QQPlot(pvals[j].as_matrix(), plt_title=plttitle, savepath=outpath)

    if save:
        savename = outpath + savename + '_correlations.p'
        pickle.dump([corr, pvals], open(savename, 'wb'))

    return corr, pvals


def check_k_range(data, cluster_sizes, iterations, savename):
    for k in cluster_sizes:
        print 'k= :', k
        index_pairs = defaultdict(int)
        conservation = []
        for repeat in range(2):
            km = KMeans(n_clusters=k)
            km.fit(data)
            l = km.labels_
            for i in range(k):
                pairs = set(itertools.combinations(np.where(l == i)
                            [0].tolist(), 2))
                for pair in pairs:
                    index_pairs.__getitem__(pair)
                    index_pairs[pair] += 1
        # conserved = sum(index_pairs.values()) /
        # float((len(index_pairs.keys()) * 2))
        conserved = sum([x >= 2 for x in index_pairs.values()]) /\
            float(len(index_pairs.keys()))
        print conserved
        conservation.append(conserved)

    pickle.dump(conservation, open(savename, 'wb'))
    print conservation
    return conservation


def iterative_clean(p, clean_components, regress=None, transpose=False,
                    odir='./', title='', pltfunc=PCPlot, **kwargs):
    """
    p is a preprocessing object,
    clean components is the number of PCs we want to remove
    clusters is the number of clusters to label datapoints on PC plot with
    scales the data before runnning it through PCPlot
    """
    for i in range(clean_components+1):
        c = np.arange(i)
        print 'Cleaning PCs: ', c
        clean = p.clean(components=c, regress_out=regress, update_data=False)
        if transpose:
            clean = clean.T
        clean = scale(clean)

        plt_title = title + str(c)
        pltfunc(data=clean, title=plt_title, **kwargs)

def get_common_set(list1, list2):
    """ for two nparrays of labels
    return an np array of the intersection set of the two
    and indexes into the two original arrays in the order they
    appear in the output array
    """
    common = np.empty(0)
    list1_indices = np.empty(0)
    list2_indices = np.empty(0)
    for item in list1:
        if (item in list2) and (item not in common):
            common = np.append(common, item)
            list1_indices = np.append(list1_indices,
                                      np.where(list1 == item)[0])
            list2_indices = np.append(list2_indices,
                                      np.where(list2 == item)[0])
    return common, list1_indices.astype(int), list2_indices.astype(int)



def make_config_string(config, org_params):
    string = ''
    for key in org_params:
        value = config[key]
        if not isinstance(value, float):
            string += key + '=' + str(value) + ','
        else:
            string += key + '=' + '{:.2e}'.format(value) + ','
    string = string[:-1]
    return string
