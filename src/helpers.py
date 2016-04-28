import matplotlib.pyplot as plt
import numpy as np


def plot_2D(data, color_labels=None, x=0, y=1, xlabel='X', ylabel='Y', title='2-Dimensional Plot'):
    """
    Returns subplot of data plotted over the given axes
    TODO: make it return subplots so we dont need to use new figures each time
    """
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
    return fig


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

def export_clusters(cluster_labels, name_label, savename='../data/clustered_genes.csv'):
    num_clusters = np.bincount(cluster_labels).size
    max_in_cluster = np.max(np.bincount(cluster_labels))

    out = np.empty([max_in_cluster, num_clusters], dtype=object)
    for i in range(0,num_clusters):
        new = name_label[np.where(cluster_labels == i)[0]]
        out[0:(new.size),i] = new
        out[(new.size)::, i] = ''

    np.savetxt(savename, out, delimiter='\t', newline='\n', fmt='%s')
    return out

