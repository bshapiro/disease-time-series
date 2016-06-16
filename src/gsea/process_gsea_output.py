from glob import glob
from optparse import OptionParser


def get_sigs(output_directory, p_value_threshold=0.05, fdr_threshold=0.1):
    sig_dict = {}
    files = glob(output_directory + "*")
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
        if significant != []:
            sig_dict[filename] = significant
    return sig_dict


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--fdr", dest="fdr_threshold",
                      help="FDR threshold")
    parser.add_option("-p", "--p-value", dest="p_value_threshold", default=0.05,
                      help="p-value threshold")
    parser.add_option("-d", "--d", dest="gsea_output_directory",
                      help="directory for GSEA output files")

    (options, args) = parser.parse_args()

    sig_dict = get_sigs(options.gsea_output_directory, options.p_value_threshold, options.fdr_threshold)
    print "Significant enrichments:"
    for key, value in sig_dict.items():
        print key
        for sig in value:
            print '\t', sig[0], sig[-3], sig[-2], sig[-1]
