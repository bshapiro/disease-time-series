from preprocessing import *
from representation import *
from tools.helpers import *
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

mc_data = '../data/my_connectome/GSE58122_varstab_data_prefiltered_geo.txt'
mc_genes = '../data/my_connectome/mc_geneset.p'
fd = ['../data/my_connectome/rin.p']
ops = ['>']
th = [6]
rin = pickle.load(open(fd[0]))
goodrin = rin[np.where(rin > th[0])[0]].reshape(-1,1)

gc = Preprocessing(mc_data, transpose=False, filter_data=fd, operators=ops, thresholds=th, clean_components=[0,1,2], sample_labels=mc_genes, regress_out=goodrin, odir='explore')

gcdf = pd.DataFrame(data=gc.data, index=gc.s)

track = pd.read_csv('../data/my_connectome/tracking_data.txt', sep='\t', na_values='.', header=0, index_col=0)
track = track[track.loc[:,'rna:rin'] > 6]
gcdf.columns = track.index

targets = np.array(['blood:eo', 'blood:ba', 'blood:ba', 'blood:eo', 'blood:hgb', 'blood:ly', 'blood:mch', 'blood:mchc', 'blood:mcv', 'blood:mo', 'blood:mpv', 'blood:ne', 'blood:plt', 'blood:rbc', 'blood:wbc', 'morning:Pulse', 'morning:Sleepquality', 'morning:Soreness', 'panas:fatigue', 'prevevening:Alcohol', 'prevevening:Guthealth', 'prevevening:Howmuchdidtinnitusbotheryoutoday?', 'prevevening:Psoriasisseverity', 'prevevening:Stress', 'prevevening:Timespentoutdoors', 'sameevening:Alcohol', 'sameevening:Guthealth', 'sameevening:Howmuchdidtinnitusbotheryoutoday?', 'sameevening:Psoriasisseverity', 'sameevening:Stress', 'sameevening:Timespentoutdoors', 'weather:precip', 'weather:temphi', 'weather:templo', 'weight', 'zeo:timeInDeep', 'zeo:timeInLight', 'zeo:timeInRem', 'zeo:totalZ'])

kmeans_assignments = Representation(gc.data, 'kmeans').getRepresentation()
#cluster_labels = ['c'+str(i) for i in range(kmeans_assignments[1].shape[0])]
#cluster_df = pd.DataFrame(data=kmeans_assignments[1], index=cluster_labels, columns=track.index)

#results = associate(gcdf, track, targets2=targets, outpath='../results/')
#pickle.dump(results, open('../results/genewise_spearman_correlation_p_values.p', 'wb'))

#cluster_results = associate(cluster_df, track, targets2=targets, outpath='../results/k115_')

#sleep = track.loc[:,('zeo:timeInLight','zeo:timeInDeep','zeo:timeInRem')].as_matrix()
#sleep = np.nan_to_num(sleep)
#sleep = sleep / sleep.sum(1).reshape(-1,1)

#notnan = np.invert(np.isnan(sleep[:,0]))

#sleep_prop_p = np.empty((gc.data.shape[0], 3))

"""
for i in range(gc.data.shape[0]):
    for j in range(3):
        sleep_prop_p[i,j] = sp.stats.linregress(gc.data[i,notnan], sleep[notnan,j])[3]

QQPlot(sleep_prop_p.flatten())
sleep_prop_significant = benjamini(sleep_prop_p, 0.05)
"""

"""
sleepq = track.loc[:,'zeo:zq'].as_matrix()
notnan = np.invert(np.isnan(sleepq))
sleepq_p = np.empty(gc.data.shape[0])
light_p = np.empty(gc.data.shape[0])
deep_p = np.empty(gc.data.shape[0])
rem_p = np.empty(gc.data.shape[0])
for i in range(gc.data.shape[0]):
    sleepq_p[i] = sp.stats.spearmanr(gc.data[i,notnan],sleepq[notnan])[1]
    light_p[i] = sp.stats.spearmanr(gc.data[i,notnan],sleep[notnan, 0])[1]
    deep_p[i] = sp.stats.spearmanr(gc.data[i,notnan],sleep[notnan, 1])[1]
    rem_p[i] = sp.stats.spearmanr(gc.data[i,notnan],sleep[notnan, 2])[1]

QQPlot(sleepq_p.flatten())
QQPlot(light_p.flatten())
QQPlot(deep_p.flatten())
QQPlot(rem_p.flatten())

sleepq_sig = benjamini(sleepq_p, 0.05)
rem_sig = benjamini(rem_p, 0.05)
light_sig = benjamini(light_p, 0.05)
deep_sig = benjamini(deep_p, 0.05)
"""
            
