from glob import glob
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-f", "--fdr", dest="fdr_threshold",
                  help="FDR threshold")
parser.add_option("-p", "--p-value", dest="p_value_threshold", default=0.05,
                  help="p-value threshold")
parser.add_option("-d", "--d", dest="gsea_output_directory",
                  help="directory for GSEA output files")

(options, args) = parser.parse_args()


file_dict = {}
files = glob(options.gsea_output_directory + "*.csv")
for filename in files:
    f = open(filename)
    lines = f.readlines()[1:]
    significant = []
    for line in lines:
        items = line[:-1].split(',')
        bp = items[0]
        p_value = items[1]
        fdr = items[2]
        if float(fdr) < float(options.fdr_threshold) and float(p_value) < float(options.p_value_threshold):
            significant.append(items)
    if significant != []:
        file_dict[filename] = significant

for key, value in file_dict.items():
    print key
    for sig in value:
        print '\t', sig
