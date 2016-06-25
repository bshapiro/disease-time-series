import addpath
import numpy as np
from pickle import load
from src.tools.accessioncodes import *

gene_acc = load(open('ensemble_gene_codes.p'))

server = 'http://www.ensembl.org/biomart'
dataset = 'hsapiens_gene_ensembl'
filters = ['ensembl_gene_id']
atts = ['external_gene_name']

# other attributes that may be useful
# ['external_gene_source','ensembl_gene_id','ensembl_transcript_id','ensembl_peptide_id']

searcher = BioMartSearch(gene_acc[0:10], server, dataset, filters, atts)

