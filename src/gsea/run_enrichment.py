import sys
import os
from gsea import gsea
import pickle


def run_enrichment(clusters, all_genes, geneset_name, save_name):
    geneset_title = os.path.splitext(os.path.basename(geneset_name))[0]

    all_genes = set(all_genes)

    for cluster, test_genes in clusters.iteritems():
        test_genes = set(test_genes)
        test_out = gsea(test_genes, all_genes, geneset_name)
        """ Save result to save_name"""
        try:
            f = open(save_name + '_' + geneset_title + '_'+cluster, 'w')
        except:
            directory = '/'.join(save_name.split('/')[:-1])
            print "Creating directory...", directory
            os.mkdir(directory)
            f = open(save_name + '_' + geneset_title + '_'+cluster, 'w')
        print >>f, "\ttest_inset\ttest_notinset\tbackground_inset\tbackground_notinset\toddsratio\tpvalue\tbonferroni-adjusted\ttestgenes_in_set"
        for i in test_out:
            print >>f, i, '\t', '\t'.join(str(j) for j in test_out[i])
        f.close()

if __name__ == "__main__":
    file_name = sys.argv[1]  # File with dictionary of clusters {cluster_name : genes}
    background_file_name = sys.argv[2]  # File with background genes (one gene per line)
    geneset_name = sys.argv[3]  # filepath+filename of a geneset downloaded from msigdb. Need not parse msigdb file. Script takes care of parsing
    save_name = sys.argv[4]  # filepath+name for output file

    """ Read test genes"""
    fh = open(file_name, 'r')
    clusters = pickle.load(fh)
    fh.close()

    """ Read background genes"""
    fh = open(background_file_name, 'r')
    all_genes = pickle.load(fh)
    all_genes = all_genes.astype(list)
    fh.close()

    run_enrichment(clusters, all_genes, geneset_name, save_name)
