import matplotlib.pyplot as plt
from config import config
import numpy as np


def histogram_analysis(matrix, view_name):
    index = 0
    for column in matrix.T:
        print column
        y, binEdges = np.histogram(column, bins=100000)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        plt.plot(bincenters, y, '-', label='Timestep ' + str(index))
        index += 1
    plt.legend()
    plt.xlim(0, 100)
    plt.savefig(config['project_root'] + 'data/histogram_view=' + view_name + '.png')
    plt.clf()
