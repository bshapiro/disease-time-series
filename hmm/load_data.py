import addpath
from src.preprocessing import Preprocessing, load_file
from datetime import datetime
import pickle
import pandas as pd
import numpy as np

mc_data = '../data/my_connectome/GSE58122_varstab_data_prefiltered_geo.txt'
mc_genes = '../data/my_connectome/mc_geneset.p'
track_data = '../data/my_connectome/tracking_data.txt'
fd = ['../data/my_connectome/rin.p']
ops = ['>']
th = [6]
rin = pickle.load(open(fd[0]))
goodrin = rin[np.where(rin > th[0])[0]].reshape(-1, 1)


def load_data(m=None, seed=None):

    # seed random number generation
    if seed is not None:
        np.random.seed(seed)

    data, rl, cl = load_file(mc_data, 'tsv', True, True)
    gc = Preprocessing(data, rl, cl, transpose=True)
    gc.filter((rin, '>', 6, 0))

    gc.clean(components=[0, 1, 2], regress_out=[(goodrin, 0)],
             update_data=True, scale_out=False)
    gc.transpose()
    gc.scale(1)
    if m is not None:
        indices = np.arange(gc.data.index.size)
        indices = np.random.permutation(indices)
        indices = indices[:m]
        gc.data = gc.data.iloc[indices, :]

    track_data = '../data/my_connectome/tracking_data.txt'
    track = pd.read_csv(track_data, sep='\t', na_values='.', header=0,
                        index_col=0)
    track = track[track.loc[:, 'rna:rin'] > 6]

    metab_file = '../data/my_connectome/metabolomics_raw_data.csv'
    metab = pd.read_csv(metab_file, sep='\t', index_col=0, header=4,
                        usecols=[i for i in range(6, 53)].append(0),
                        skiprows=[5, 6, 7])
    metab = metab.iloc[:, 5:]

    date_to_sample = dict(zip(track['date'].as_matrix(), track.index.tolist()))

    metab_dates = metab.axes[1].values.tolist()
    metab_dates = [datetime.strptime(i, "%m/%d/%Y") for i in metab_dates]
    metab_dates = [datetime.strftime(i, "%Y-%m-%d") for i in metab_dates]

    metab = pd.DataFrame(metab.as_matrix(), index=metab.axes[0],
                         columns=metab_dates)

    metab_samples = []
    usable_dates = []
    for date in metab_dates:
        if date in date_to_sample:
            usable_dates.append(date)
            metab_samples.append(date_to_sample[date])

    metab = metab.loc[:, usable_dates]
    metab = pd.DataFrame(metab.as_matrix(), index=metab.axes[0],
                         columns=metab_samples)

    metab = metab.iloc[:106, :]

    metab_id_file = '../data/my_connectome/metab_ids.txt'
    metab_id_table = np.genfromtxt(metab_id_file, delimiter='\t',
                                   skip_header=1, dtype=str)

    metab = metab.iloc[np.where(metab_id_table[:, 2] != 'Not Found')[0], :]
    kegg_ids = metab_id_table[np.where(metab_id_table[:, 2]
                              != 'Not Found')[0], 2]

    mt = Preprocessing(metab.as_matrix(), kegg_ids, metab.columns.values,
                       transpose=False)
    mt.log_transform(0)
    mt.scale(1)

    return gc, mt, track
