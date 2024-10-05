from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
from logging import raiseExceptions
import pandas as pd
import subprocess
import sys
import os
from pathlib import Path
from scipy.stats import hypergeom
import numpy as np
from scipy.stats import fisher_exact
import logging
from variant_driven_ASE import collect_info_one_gene, collect_info_common_SNPs, calculate_2sidedfisherexact_haplotype
"""
write to file line by line
"""


# The probability that we would observe this or an even more imbalanced ratio by chance is about 3.5%. A commonly used significance level is 5%–if we adopt that, we can therefore conclude that our observed imbalance is statistically significant; whales prefer the Atlantic while sharks prefer the Indian ocean."
# python annotate_ASE.py /data2/genes/find_novel_imprinted_gene_totalcount10_top50.tsv 10 /data2/genes/find_novel_imprinted_gene_totalcount10_top50_annotated.tsv

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.error("Gene candidate argument missing.")
        sys.exit(1)

    in_file = sys.argv[1]
    cutoff_totalcount = sys.argv[2]
    out_file = sys.argv[3]

    data_dir = "/data2/1000Genome"
    save_dir = "/data2/genes"

    ####################################### parameters setting
    input_file = pd.read_csv(in_file, sep="\t", header=0)
    gene_list = input_file["geneID"].tolist()  # ["ENSG00000164308.12"]

    ####################################### write to a file line by line
    with open(out_file, "w") as out:
        out.write(
            "geneID\tGene.name\ttotal_nInd_R\ttotal_nInd\textremeASE_nInd\textremeASE_percent\tASE_nInd\tASE_percent\tcommon_SNP\tcommon_SNP_count\tcommon_snp_percent\tfishexactP\n"
        )
        for i in range(0, input_file.shape[0]):
            line = input_file.iloc[i]
            genesymbol = line["geneID"]
            genename = line["gene_name"]
            total_individual = line["total_nind"]
            extremeASE_nInd = line["num_extremeASE"]
            extremeASE_percent = round(extremeASE_nInd/total_individual,3)
            ASE_nInd = line["num_ASE"]
            ASE_percent = round(ASE_nInd/total_individual,3)
            
            #extremeASE_nAncesrty = line["extremeASE_ancestry"]

            common_SNP = collect_info_one_gene(cutoff_totalcount, data_dir, genesymbol, genename, False)
            common_SNP_data_snp, common_SNP_data_snp_list = collect_info_common_SNPs(common_SNP, False)
            
            data = common_SNP
            total_ind = len(pd.unique(common_SNP["sample"]))
            chrN = data["chr"].iloc[0]
            mostcomon_snp = common_SNP_data_snp_list[0]
            the_common_snp = chrN + ":" + str(mostcomon_snp)
            common_snp_count = common_SNP_data_snp.to_dict()[mostcomon_snp]
            percent = round(common_snp_count / total_individual, 3)

            (
                pval,
                ASE_REF,
                ASE_ALT,
                noASE_REF,
                noASE_ALT,
            ) = calculate_2sidedfisherexact_haplotype(mostcomon_snp,data)

            out.write(
                str(genesymbol)
                + "\t"
                + genename
                + "\t"
                + str(total_individual)
                + "\t"
                + str(total_ind)
                + "\t"
                + str(extremeASE_nInd)
                + "\t"
                + str(extremeASE_percent)
                + "\t"
                + str(ASE_nInd)
                + "\t"
                + str(ASE_percent)
                + "\t"
                + str(the_common_snp)
                + "\t"
                + str(common_snp_count)
                + "\t"
                + str(percent)
                + "\t"
                + str(pval)
                + "\n"
            )
    out.close()
