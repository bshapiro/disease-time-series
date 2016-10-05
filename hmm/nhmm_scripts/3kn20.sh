python nstate_hmm.py 20 -3 3 0.5 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/ 5
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/0/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/1/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/2/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/3/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/4/
python hmm_agglomerative_path_clustering.py 3k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/3kn10/4/
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/ 10 /scratch0/battle-fs1/my_connectome/profilehmm/3kn20_agglom/k10
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/ 20 /scratch0/battle-fs1/my_connectome/profilehmm/3kn20_agglom/k20
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/ 30 /scratch0/battle-fs1/my_connectome/profilehmm/3kn20_agglom/k30
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/3kn20/ 60 /scratch0/battle-fs1/my_connectome/profilehmm/3kn20_agglom/k60