def export_clusters(cluster_labels, gene_set, savename='../out/clustered_genes.p'):
    num_clusters = np.bincount(cluster_labels).size
    cluster_dict = {}
    for i in range(num_clusters):
        cluster_label = 'c' + str(i)
        if cluster_dict.get(cluster_label) is None:
            genes = gene_set[np.where(cluster_labels == i)[0]]
            cluster_dict[cluster_label] = genes
        
    pickle.dump(cluster_dict, open(savename, 'w'))
    #sio.savemat(savename, cluster_dict)

def plot_2D(data, color_labels=None, x=0, y=1, xlabel='X', ylabel='Y', title='2-Dimensional Plot'):
    Returns subplot of data plotted over the given axes
    TODO: make it return subplots so we dont need to use new figures each time
    fig = plt.figure()
    x_vals = data[:, x]
    y_vals = data[:, y]
    colors = label_coloring(color_labels)
    if colors is not None:
        plt.scatter(x_vals, y_vals, figure=fig, s=100, c=colors)
    else:
        plt.scatter(x_vals, y_vals, figure=fig, s=100)
    plt.title(title, figure=fig)
    plt.xlabel(xlabel, figure=fig)
    plt.ylabel(ylabel, figure=fig)

def label_coloring(labels):
    """
    returns a coloring where each unique label gets its own color
    """
    if labels is None:
        return None
    num_colors = np.bincount(labels).size
    color_vals = np.linspace(0, 1, num_colors, endpoint = True)
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
    SV = np.zeros((n,n))
    SV[np.diag_indices(n)] = pca.explained_variance_
    SV = np.sqrt(SV)
 
    # X = U SV W.T, where the columns of W are are eigenvectors/PCs
    # since rows of pca.components_ are PCs pca.components_ = W.T
    W = pca.components_.T
    # U = X W SV^-1, since W W.T = I
    U = X.dot(W).dot(np.linalg.inv(SV))
    return U, SV, W


    def regress_out(self,ind, X=None, axis=None):
        if X is None:
            X = self.X
        if axis is None:
            # try and figure out what axis based on the dimensions of independent variable
            n = ind.shape[0]
            if X.shape[0] == X.shape[1]:
                # this is an ambigous case
                print >> sys.stderr, 'Ambiguous dimensions, please specify axis'
                return
            if X.shape[0] == n:
                axis = 0
            elif X.shape[1] == n:
                axis = 1
            else:
                print >> sys.stderr, 'Mismatched dimensions'
                
        print 'Axis: ', axis
        if axis == 1:
            X = np.transpose(X)
        linreg = LinearRegression()
        linreg.fit(ind, X)
        new_data = X - linreg.predict(ind)
        # add back the intercept
        new_data = np.apply_along_axis(np.add, 1, new_data, linreg.intercept_)
        if axis == 1:
              X = np.transpose(X)
              new_data = np.transpose(new_data)
        return new_data, linreg

