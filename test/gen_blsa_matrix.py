from glob import glob
import numpy as np
import pandas as pd
from pickle import load, dump

trascript_dict = load(open('blsa_transcript_dict.p', 'rb'))
transcript_names = transcript_dict.values()

samples = set([name[46:50] for name in glob("/scratch0/battle-fs1/blsa-pilot/donor_samples/*")])

data = []
for sample in samples:
    files = glob(sample + '*')
    read = []
    for i, file in enumerate(files):
        read.append(np.genfromtxt(file, delimiter='\t', usecols=[10], skip_header=1))
    pbmc = read[0]
    pbmc.reshape(-1,1)
    for i in range(1, len(read)):
        pbmc = np.concatenate(pbmc, read[i].reshape(-1,1))
    pbmc = pbmc.mean(1)
    data.append(pbmc)
    columns.append(sample)

fd = pd.DataFrame(data=data, index=transcript_names, columns=samples)







    for filename in files:
        f = open(filename)
        lines = f.readlines()[4:]
        significant = []
        for line in lines:
            items = line[:-1].split('\t')
            p_value = items[6]
            fdr = items[7]
            if float(fdr) < float(fdr_threshold) and float(p_value) < float(p_value_threshold):
                significant.append(items)
        filename_suffix = filename.split('/')[-1]
        if significant != []:
            sig_dict[filename_suffix] = significant
    return sig_dict