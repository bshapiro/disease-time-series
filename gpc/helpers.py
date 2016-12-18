from config import config
from os import makedirs
import pandas as pd
import numpy as np


def generate_output_dir():
    directory = ''
    directory += config['views'] + '/'
    directory += config['init'] + '/'
    directory += config['kernel'] + '/'
    if config['differential_transform']:
        directory += 'differential/'
    else:
        directory += 'raw/'
    directory += config['dir'] + str(config['k']) + '/'
    if config['views'] is not 'single':
        directory += 'strength' + str(config['strength']) + '/'
    else:
        directory += config['dataset'] + '/'
    try:
        makedirs(directory)
    except:
        pass
        # print 'Output directory already exists.'
    return directory


def flatten_tuple_list(tuple_list):
    flat_list = []
    for a_tuple in tuple_list:
        flat_list.extend(list(a_tuple))
    return flat_list


def pair_dict(pairs):
    pair_dict = {}
    for pair in pairs:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    return pair_dict


def load_te():
    """
    Load TE and rearrange in the correct gene order.
    """
    genes = [item[0] for item in pd.read_csv(open('../data/myeloma/genes.csv'), header=None).values.tolist()]
    te = pd.read_csv(open('../data/myeloma/te.csv'), sep=',', header=0)
    new_te = []
    for gene in genes:
        new_te.append(te[te['GeneID'] == gene].values.tolist()[0][1:])
    te = np.log2(np.asarray(new_te))
    return te


def load_myeloma_paper_clusters():
    genes = [item[0] for item in pd.read_csv(open('../data/myeloma/genes.csv'), header=None).values.tolist()]
    upreg = [item[0] for item in pd.read_csv(open('../data/myeloma/upreg.csv'), header=None).values.tolist()]
    downreg = [item[0] for item in pd.read_csv(open('../data/myeloma/downreg.csv'), header=None).values.tolist()]
    stable = [item[0] for item in pd.read_csv(open('../data/myeloma/stable.csv'), header=None).values.tolist()]
    tedown = [item[0] for item in pd.read_csv(open('../data/myeloma/tedown.csv'), header=None).values.tolist()]
    teup = [item[0] for item in pd.read_csv(open('../data/myeloma/teup.csv'), header=None).values.tolist()]

    upreg_indices = []
    downreg_indices = []
    stable_indices = []
    tedown_indices = []
    teup_indices = []

    for gene in tedown:
        tedown_indices.append(genes.index(gene))
    for gene in downreg:
        downreg_indices.append(genes.index(gene))
    for gene in teup:
        teup_indices.append(genes.index(gene))
    for gene in stable:
        stable_indices.append(genes.index(gene))
    for gene in upreg:
        upreg_indices.append(genes.index(gene))

    return upreg_indices, downreg_indices, stable_indices, tedown_indices, teup_indices
