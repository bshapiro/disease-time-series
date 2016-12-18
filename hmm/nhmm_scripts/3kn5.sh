cd /home/ktayeb1/disease-time-series/hmm
python nstate_hmm.py 5 -2 2 0.5 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/ 5
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/0/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/1/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/2/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/3/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/4/
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/ 10 /scratch0/battle-fs1/my_connectome/profilehmm/3kn5_agglom/k10
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/ 20 /scratch0/battle-fs1/my_connectome/profilehmm/3kn5_agglom/k20
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/ 30 /scratch0/battle-fs1/my_connectome/profilehmm/3kn5_agglom/k30
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/3kn5/ 60 /scratch0/battle-fs1/my_connectome/profilehmm/3kn5_agglom/k60
