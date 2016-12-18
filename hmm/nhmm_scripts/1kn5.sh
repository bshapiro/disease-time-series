cd /home/ktayeb1/disease-time-series/hmm
python nstate_hmm.py 5 -2 2 0.5 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn5/ 5
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn5/0/
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn5/1/
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn5/2/
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn5/3/
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn5/4/
