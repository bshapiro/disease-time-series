import addpath
import numpy as np
from pickle import load, dump
from src.tools.accessioncodes import *

# gene_acc = load(open('ensembl_genes_codes.p'))
g, t = load(open('ipop_ids.p'))
transcript_ids = [transcript.split(',')[0] for transcript in t[1:]]
server = 'http://www.ensembl.org/biomart'
dataset = 'hsapiens_gene_ensembl'
"""
filters = ['ensembl_transcript_id',]
atts = ['external_gene_name']

# other attributes that may be useful
# ['external_gene_source','ensembl_gene_id','ensembl_transcript_id','ensembl_peptide_id']

searcher = BioMartSearch(gene_acc, server, dataset, filters, atts)
searcher.bulksearch(filters[0])
dump(searcher.dictionary, open('blsa_transcript_dict.p', 'wb'))
"""

filters = ['ucsc']
atts = ['external_gene_name']

# other attributes that may be useful
# ['external_gene_source','ensembl_gene_id','ensembl_transcript_id','ensembl_peptide_id']

searcher = BioMartSearch(transcript_ids, server, dataset, filters, atts)
searcher.bulksearch(filters[0])
dump(searcher.dictionary, open('blsa_transcript_dict.p', 'wb'))