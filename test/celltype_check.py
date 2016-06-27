import addpath
from src.preprocessing import *
from src.representation import *
from src.tools.helpers import *
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import load, dump

mc_data = '../data/my_connectome/GSE58122_varstab_data_prefiltered_geo.txt'
mc_genes = '../data/my_connectome/mc_geneset.p'
track_data = '../data/my_connectome/tracking_data.txt'
fd = ['../data/my_connectome/rin.p']
ops = ['>']
th = [6]
rin = pickle.load(open(fd[0]))
goodrin = rin[np.where(rin > th[0])[0]].reshape(-1,1)

data, rl, cl = load_file(mc_data, '\t', True, True)
gc = Preprocessing(data, rl, cl)

gc.reset()
gc.data = gc.data.astype(float)
gc.filter((rin, '>', 6, 1))

# gc.clean(components=3, regress_out=[(goodrin, 1), ])

track = pd.read_csv(track_data, sep='\t', na_values='.', header=0, index_col=0)
track = track[track.loc[:,'rna:rin'] > 6]

# list of genes correlated with lymphocyte levels
candidates = ['RPS24', 'RPL21', 'MDS1', 'UQCRB', 'RPL11', 'HSPC016', 'RPL35',
              'RPS14', 'RPS6KA2', 'PCP2', 'RPL27', 'RPL31', 'SERPINB6',
              'LEREPO4', 'TGT', 'RPS21', 'RPS6', 'TPI1', 'LSM3', 'E2IG5',
              'CYP17A1', 'AMSH-LP', 'COX6C', 'KIAA0515', 'RPL32', 'RPL23',
              'ATF4', 'RPS3A', 'JUN', 'SESTD1', 'COX7B', 'MALAT1', 'LOC134121',
              'EEF2', 'SLC19A1', 'FGF9', 'TPP2', 'NSUN4', 'ATXN2L']

blood_panel = np.array(['blood:eo', 'blood:ba', 'blood:hgb', 'blood:ly',
                        'blood:mch', 'blood:mchc', 'blood:mcv', 'blood:mo',
                        'blood:mpv', 'blood:ne', 'blood:plt',
                        'blood:rbc', 'blood:wbc'])
pbmc = np.array(['blood:mo', 'blood:ly'])

# whittle it down to genes that we have
lymphocyte_genes = []
for gene in candidates:
    if gene in gc.data.index:
        lymphocyte_genes.append(gene)

lymphocyte_genes = np.array(lymphocyte_genes)

blsa_transcripts = load(open('blsa_transcript_dict.p'))
blsa_df = pd.read_pickle('pbmc_dataframe.p')

common_genes = np.array(list(set(gc.data.index.values).intersection(set(blsa_df.index.values))))

meth = 'spearman'
# on raw data
c, p = sp.stats.spearmanr(gc.data.as_matrix())

# regress rin and top 3 PCs
labels = gc.data.columns.values
for i in range(4):
    pcs = np.arange(i)
    clean = gc.clean(components=pcs, regress_out=[(goodrin, 1), ], update_data=False)
    # clean = gc.clean(components=pcs, update_data=False)

    # clean_df = pd.DataFrame(clean, index=gc.data.index, columns=gc.data.columns)
    # savename = '../results/celltype_check/rin+pc' + str(c)
    # c, p = associate(clean_df, track, targets1=lymphocyte_genes,
    #                        targets2=pbmc,  method=meth, outpath=savename)
    c, p = sp.stats.spearmanr(clean)
    p += 1e-20
    p = -1 * np.log10(p)
    plt_title = 'rin+' + str(pcs) + 'Spearman p.png'
    HeatMap(p, labels, labels, title=plt_title)