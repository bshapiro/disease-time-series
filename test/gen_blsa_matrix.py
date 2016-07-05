from glob import glob
import numpy as np
import pandas as pd
from pickle import load, dump

transcript_dict = load(open('blsa_transcript_dict.p', 'rb'))
transcript_names = transcript_dict.values()

samples = set([name[46:51] for name in glob("/scratch0/battle-fs1/blsa-pilot/donor_samples/*")])
print samples

data = []
for sample in samples:
    files = glob('/scratch0/battle-fs1/blsa-pilot/donor_samples/' + sample + '*')
    read = []
    for i, file in enumerate(files):
        read.append(np.genfromtxt(file, delimiter='\t', usecols=[10], skip_header=1))
    pbmc = read[0]
    pbmc = pbmc.reshape(-1,1)
    for i in range(1, len(read)):
        pbmc = np.concatenate((pbmc, read[i].reshape(-1,1)), 1)
    pbmc = pbmc.mean(1)
    data.append(pbmc)

fd = pd.DataFrame(data=np.array(data).T, index=transcript_names, columns=samples)
