cd /home/ktayeb1/disease-time-series/hmm
python nstate_hmm.py 10 -3 3 0.5 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/ 5
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/0/
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/1/
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/2/
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/3/
python hmm_agglomerative_path_clustering.py 1k_genes.p /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/4/
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/ 10 /scratch0/battle-fs1/my_connectome/profilehmm/1kn10_agglom/k10
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/ 20 /scratch0/battle-fs1/my_connectome/profilehmm/1kn10_agglom/k20
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/ 30 /scratch0/battle-fs1/my_connectome/profilehmm/1kn10_agglom/k30
python hmm_agglomerative_cluster_analysis.py /scratch0/battle-fs1/my_connectome/profilehmm/1kn10/ 60 /scratch0/battle-fs1/my_connectome/profilehmm/1kn10_agglom/k60