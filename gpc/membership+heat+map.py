import pickle
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
import sys

# In[30]:

directory = sys.argv[1]


# In[31]:

genes = [item[0] for item in pd.read_csv(open('../data/myeloma/genes.csv'), header=None).values.tolist()]
upreg = set([item[0] for item in pd.read_csv(open('../data/myeloma/upreg.csv'), header=None).values.tolist()])
downreg = set([item[0] for item in pd.read_csv(open('../data/myeloma/downreg.csv'), header=None).values.tolist()])
stable = set([item[0] for item in pd.read_csv(open('../data/myeloma/stable.csv'), header=None).values.tolist()])
tedown = set([item[0] for item in pd.read_csv(open('../data/myeloma/tedown.csv'), header=None).values.tolist()])
teup = set([item[0] for item in pd.read_csv(open('../data/myeloma/teup.csv'), header=None).values.tolist()])
paper_clusters = [upreg, downreg, stable, tedown, teup]
memberships = pickle.load(open(directory + 'memberships.dump'))


# In[32]:

membership_f = open(directory + 'memberships.txt', 'w')
membership_string = ','.join([str(memberships[key]) for key in sorted(memberships.keys())])
membership_f.write(membership_string)
membership_f.close()


# In[33]:

clusters = {}
if len(memberships) != 2:
    for gene_index, cluster_id in memberships.items():
        if clusters.get(cluster_id) is None:
            clusters[cluster_id] = []
        clusters[cluster_id].append(genes[gene_index])
else:
    for dataset_name, dataset in memberships.items():
        for gene_index, cluster_id in dataset.items():
            if clusters.get(cluster_id) is None:
                clusters[cluster_id] = []
            clusters[cluster_id].append(genes[gene_index])


# In[34]:

heat_map = []
for cluster_id, cluster in clusters.items():
    print cluster_id
    overlaps = []
    for paper_cluster in paper_clusters:
        overlaps.append(len(list(paper_cluster & set(cluster))) / float(len(paper_cluster)))
    heat_map.append(overlaps)
print heat_map


# In[35]:

fig, ax = plt.subplots()
data = np.asarray(heat_map)
ax = sns.heatmap(data, cmap=plt.cm.Blues, linewidths=.1)
# set the x-axis labels on the top
ax.xaxis.tick_top()
# rotate the x-axis labels
plt.xticks(rotation=45)
ax.invert_yaxis()
# get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
fig = ax.get_figure()
# specify dimensions and save
fig.set_size_inches(15, 20)
ax.set_xticklabels(['Upreg', 'Downreg', 'Stable', 'TE-up', 'TE-down'], minor=False)
plt.show()
