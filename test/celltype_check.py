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
goodrin = rin[np.where(rin > th[0])[0]].reshape(-1, 1)

data, rl, cl = load_file(mc_data, 'tsv', True, True)
gc = Preprocessing(data, rl, cl)
# gc.data = gc.data.astype(float)
gc.filter((rin, '>', 6, 1))


# gc.clean(components=3, regress_out=[(goodrin, 1), ])

track = pd.read_csv(track_data, sep='\t', na_values='.', header=0, index_col=0)
track = track[track.loc[:,'rna:rin'] > 6]

# list of genes correlated with lymphocyte levels
# candidates = ['RPS24', 'RPL21', 'MDS1', 'UQCRB', 'RPL11', 'HSPC016', 'RPL35',
#              'RPS14', 'RPS6KA2', 'PCP2', 'RPL27', 'RPL31', 'SERPINB6',
#              'LEREPO4', 'TGT', 'RPS21', 'RPS6', 'TPI1', 'LSM3', 'E2IG5',
#              'CYP17A1', 'AMSH-LP', 'COX6C', 'KIAA0515', 'RPL32', 'RPL23',
#              'ATF4', 'RPS3A', 'JUN', 'SESTD1', 'COX7B', 'MALAT1', 'LOC134121',
#              'EEF2', 'SLC19A1', 'FGF9', 'TPP2', 'NSUN4', 'ATXN2L']

mo_candidates = ['CD14', 'CD16', 'CD64', 'CD11b', 'CD68', 'CD163',
                 'CD11a', 'CD11b', 'CD11c', 'CD14', 'CD15', 'CD16', 'CD33',
                 'CD64', 'CD68', 'CD80', 'CD85k', 'CD86', 'CD105', 'CD107b',
                 'CD115', 'CD163', 'CD195', 'CD282', 'CD284', 'F4', 'GITRL',
                 'HLA-DR']

ly_candidates = ['CD19', 'CD3', 'CD56', 'CD25', 'CD69', 'CD3', 'CD16', 'CD56',
                 'CD2', 'CD5', 'CD19', 'CD20', 'CD21', 'CD35', 'CR2', 'CR1',
                 'CD22', 'CD23', 'CD40', 'CD45R', 'B220', 'CD69', 'CD70',
                 'CD79a', 'CD79b', 'CD80', 'CD86', 'CD93', 'CD137', 'CD138',
                 'CD252', 'CD267', 'CD268', 'CD279', 'IgD', 'IgM',
                 'CD3', 'CD4', 'CD45RA', 'CD45RB', 'CD62L', 'CD197',
                 'CD11b', 'CD11c', 'CD16', 'CD49b', 'CD56', 'CD57', 'CD69',
                 'CD94', 'CD122', 'CD158', 'CD161', 'CD244', 'CD314', 'CD319',
                 'CD328', 'CD335', 'Ly49', 'Ly108', 'CD3', 'CD4', 'CD84',
                 'CD126', 'CD150', 'CD154', 'CD185', 'CD252', 'CD278', 'CD279']

alcohol_candidates = ['OR2C3, C4orf51', 'MTRNR2L9', 'SMOX', 'DCUN1D3',
                      'CACNA1C', 'RRN3', 'CPNE2', 'KDELR1', 'PSMB10'
                      'DTD2', 'RBBP7', 'KIAA1715', 'KNTC1', 'SLC5A12',
                      'DNAJB13', 'IL13', 'UPK1B', 'CXCL2', 'HELZ']

alcohol_candidates2 = ['MTRNR1', 'ADCY1', 'ADCY8', 'AGRN', 'AVP', 'ARRB2'
                       'ADRB2', 'BDNF', 'CNX', 'GNAS', 'GFAP', 'RARB', 'CD143',
                       'NEP']

alcohol_candidates3 = ['SLC1A3', 'MDK', 'TIMP3', 'MAPT', 'ITPKA', 'WASF1'
                       'LPIN1', 'PMP22', 'APP', 'SYT1']
# http://onlinelibrary.wiley.com/doi/10.1111/j.1471-4159.2004.03021.x/full

mouse_genes = np.genfromtxt('alc_genes.csv', dtype=str)

liver_enzymes = np.array(['GOT1', 'GOT2', 'GPT' ])

pbmc = np.array(['blood:mo', 'blood:ly'])
alc = np.array(['prevevening:Alcohol', 'sameevening:Alcohol'])
# whittle it down to genes that we have
mo_genes = []
for gene in mo_candidates:
    if gene in gc.data.index:
        mo_genes.append(gene)

ly_genes = []
for gene in ly_candidates:
    if gene in gc.data.index:
        ly_genes.append(gene)

alcohol_genes = []
for gene in alcohol_candidates3:
    if gene in gc.data.index:
        alcohol_genes.append(gene)


mo_genes = np.array(mo_genes)
ly_genes = np.array(ly_genes)
alcohol_genes = np.array(alcohol_genes)
# on raw data
def raw():
    r, p = associate(gc.data, track, mo_genes, pbmc)
    r = r.abs()
    p += 1e-20
    p = -1 * np.log10(p)
    plt_title = 'Raw, Monocyte Marker Spearman Rho'
    HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
    plt_title = 'Raw, Monocyte Markers p'
    HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)

    r, p = associate(gc.data, track, ly_genes, pbmc)
    r = r.abs()
    p += 1e-20
    p = -1 * np.log10(p)
    plt_title = 'Raw, Lymphocyte Marker Spearman Rho'
    HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
    plt_title = 'Raw, Lymphocyte Markers p'
    HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)

# regress on monocyte and lymphocyte proportions
# cant do this without estimating celltype proportions firstw
"""
pcs = []
clean = gc.clean(components=pcs, regress_out=[(goodrin, 1),
                 (track['blood:mo'].as_matrix().reshape(-1, 1), 1),
                 (track['blood:ly'].as_matrix().reshape(-1, 1), 1)],
                 update_data=False)
# clean = gc.clean(components=pcs, update_data=False)

clean_df = pd.DataFrame(clean, index=gc.data.index, columns=gc.data.columns)
# savename = '../results/celltype_check/rin+pc' + str(c)
# c, p = associate(clean_df, track, targets1=lymphocyte_genes,
#                        targets2=pbmc,  method=meth, outpath=savename)
r, p = associate(clean_df, track, mo_genes, pbmc)
r = r.abs()
p += 1e-20
p = -1 * np.log10(p)
plt_title = 'rin+' + str(pcs) + ' Monocyte Marker Spearman Rho'
HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
plt_title = 'rin+' + str(pcs) + ' Monocyte Markers p'
HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)
"""

# celltype markers after regressing rin + pcs
"""
for i in range(5):
    pcs = np.arange(i)
    clean = gc.clean(components=pcs, regress_out=[(goodrin, 1), ], update_data=False)
    # clean = gc.clean(components=pcs, update_data=False)

    clean_df = pd.DataFrame(clean, index=gc.data.index, columns=gc.data.columns)
    # savename = '../results/celltype_check/rin+pc' + str(c)
    # c, p = associate(clean_df, track, targets1=lymphocyte_genes,
    #                        targets2=pbmc,  method=meth, outpath=savename)
    r, p = associate(clean_df, track, mo_genes, pbmc)
    r = r.abs()
    p += 1e-20
    p = -1 * np.log10(p)
    plt_title = 'rin+' + str(pcs) + ' Monocyte Marker Spearman Rho'
    HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
    plt_title = 'rin+' + str(pcs) + ' Monocyte Markers p'
    HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)

    r, p = associate(clean_df, track, ly_genes, pbmc)
    r = r.abs()
    p += 1e-20
    p = -1 * np.log10(p)
    plt_title = 'rin+' + str(pcs) + ' Lymphocyte Marker Spearman Rho'
    HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
    plt_title = 'rin+' + str(pcs) + ' Lymphocyte Markers p'
    HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)

# common_genes = np.array(list(set(gc.data.index.values).intersection(set(blsa_df.index.values))))
"""
def just_pcs():
    for i in range(5):
        pcs = np.arange(i)
        clean = gc.clean(components=pcs, update_data=False)
        clean_df = pd.DataFrame(clean, index=gc.data.index, columns=gc.data.columns)
        # savename = '../results/celltype_check/rin+pc' + str(c)
        # c, p = associate(clean_df, track, targets1=lymphocyte_genes,
        #                        targets2=pbmc,  method=meth, outpath=savename)
        r, p = associate(clean_df, track, mo_genes, pbmc)
        r = r.abs()
        p += 1e-20
        p = -1 * np.log10(p)
        plt_title = str(pcs) + ' Monocyte Marker Spearman Rho'
        HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
        plt_title = str(pcs) + ' Monocyte Markers p'
        HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)

        r, p = associate(clean_df, track, ly_genes, pbmc)
        r = r.abs()
        p += 1e-20
        p = -1 * np.log10(p)
        plt_title = str(pcs) + ' Lymphocyte Marker Spearman Rho'
        HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
        plt_title = str(pcs) + ' Lymphocyte Markers p'
        HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)
    return r, p

def rin_and_pcs():
    for i in range(5):
        pcs = np.arange(i)
        clean = gc.clean(components=pcs, regress_out=[(goodrin, 1), ], update_data=False)
        clean_df = pd.DataFrame(clean, index=gc.data.index, columns=gc.data.columns)
        # savename = '../results/celltype_check/rin+pc' + str(c)
        # c, p = associate(clean_df, track, targets1=lymphocyte_genes,
        #                        targets2=pbmc,  method=meth, outpath=savename)
        r, p = associate(clean_df, track, mo_genes, pbmc)
        r = r.abs()
        p += 1e-20
        p = -1 * np.log10(p)
        plt_title = 'rin+' + str(pcs) + ' Monocyte Marker Spearman Rho'
        HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
        plt_title = 'rin+' + str(pcs) + ' Monocyte Markers p'
        HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)

        r, p = associate(clean_df, track, ly_genes, pbmc)
        r = r.abs()
        p += 1e-20
        p = -1 * np.log10(p)
        plt_title = 'rin+' + str(pcs) + ' Lymphocyte Marker Spearman Rho'
        HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
        plt_title = 'rin+' + str(pcs) + ' Lymphocyte Markers p'
        HeatMap(p, p.index.values, p.columns.values, 0, 3, title=plt_title)
    return r, p

def alcohol():
    for i in range(5):
        pcs = np.arange(i)
        clean = gc.clean(components=pcs, regress_out=[(goodrin, 1), ], update_data=False)
        clean_df = pd.DataFrame(clean, index=gc.data.index, columns=gc.data.columns)
        # savename = '../results/celltype_check/rin+pc' + str(c)
        # c, p = associate(clean_df, track, targets1=lymphocyte_genes,
        #                        targets2=pbmc,  method=meth, outpath=savename)
        r, p = associate(clean_df, track, alcohol_genes, alc)
        r = r.abs()
        p += 1e-20
        p = -1 * np.log10(p)
        plt_title = 'rin+' + str(pcs) + ' Alcohol Marker rho'
        HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
        plt_title = 'rin+' + str(pcs) + ' Alcohol Marker p'
        HeatMap(p, p.index.values, p.columns.values, 0, 5, title=plt_title)

    for i in range(5):
        pcs = np.arange(i)
        clean = gc.clean(components=pcs, regress_out=None, update_data=False)
        clean_df = pd.DataFrame(clean, index=gc.data.index, columns=gc.data.columns)
        # savename = '../results/celltype_check/rin+pc' + str(c)
        # c, p = associate(clean_df, track, targets1=lymphocyte_genes,
        #                        targets2=pbmc,  method=meth, outpath=savename)
        r, p = associate(clean_df, track, alcohol_genes, alc)
        r = r.abs()
        p += 1e-20
        p = -1 * np.log10(p)
        plt_title = str(pcs) + ' Alcohol Marker rho'
        HeatMap(r, r.index.values, r.columns.values, 0, 1, title=plt_title)
        plt_title = str(pcs) + ' Alcohol Marker p'
        HeatMap(p, p.index.values, p.columns.values, 0, 5, title=plt_title)


"""
meth = 'spearman'
# on raw data
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
"""

"""
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
"""