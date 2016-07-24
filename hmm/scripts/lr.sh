cd /home/ktayeb1/disease-time-series/hmm

python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 5 3 -o ../results/final/timevariant/ -b cyc_5_3
python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 10 3 -o ../results/final/timevariant/ -b cyc_10_3
python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 15 3 -o ../results/final/timevariant/ -b cyc_15_3

python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 5 5 -o ../results/final/timevariant/ -b cyc_5_5
python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 10 5 -o ../results/final/timevariant/ -b cyc_10_5
python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 15 5 -o ../results/final/timevariant/ -b cyc_15_5

python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 5 10 -o ../results/final/timevariant/ -b cyc_5_10
python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 10 10 -o ../results/final/timevariant/ -b cyc_10_10

python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 20 3 -n 15 3 -n 10 5 -n 5 3  -o ../results/final/timevariant/ -b cyc_20_3_15_3_10_3_5_3
python gen_model_initializations.py -g 3k_genes.p -a viterbi -r 1 -k 55 -i  cycle -n 20 5 -n 15 5 -n 10 5 -n 5 5  -o ../results/final/timevariant/ -b cyc_20_5_15_5_10_5_5_5