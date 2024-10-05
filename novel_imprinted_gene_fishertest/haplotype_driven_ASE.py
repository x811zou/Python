import pandas as pd
import os
from pathlib import Path
from statsmodels.stats.multitest import multipletests
import sys
import logging
import numpy as np
from scipy.stats import fisher_exact
from variant_driven_ASE import collect_info_one_gene, collect_info_common_SNPs

"""
python haplotype_driven_ASE.py ENSG00000164308

geneID  pvals_corrected combined_SNPs   sample  ASE_gene
ENSG00000164308 1.0469791929760155e-76  96211741_96231000_96232142_96232286_96232402_96235896_96237326_96245343_96245439_96245518_96245617_96245892_96249115_96254209_96254354_96254817   NA12829 ASE
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collect_info_SNPs(save_dir, data,geneName):
    data_snp_list = data["pos"].value_counts().index.tolist()
    data_snp = data["pos"].value_counts()
    geneID = data["geneID"].iloc[0]
    data_snp.to_csv(save_dir + "/" + geneName + "/SNPs_frequency_table.tsv", sep="\t")
    return data_snp, data_snp_list


def calculate_2sidedfisherexact_variant(variant, data):
    data_filtered = data[data["pos"] == variant]
    # print(
    #     data_filtered.groupby(["ASE gene"])["compare_2allele"]
    #     .value_counts()
    #     .reindex(data_filtered.compare_2allele.unique(), fill_value=0)
    # )
    # ASE gene  compare_2allele
    # ASE       ALT                14
    #          REF                13
    # no ASE    ALT                12
    #          REF                 8
    ASE_REF = (
        data_filtered[data_filtered["ASE gene"] == "ASE"]["compare_2allele"] == "REF"
    ).sum()
    ASE_ALT = (
        data_filtered[data_filtered["ASE gene"] == "ASE"]["compare_2allele"] == "ALT"
    ).sum()
    noASE_REF = (
        data_filtered[data_filtered["ASE gene"] == "no ASE"]["compare_2allele"] == "REF"
    ).sum()
    noASE_ALT = (
        data_filtered[data_filtered["ASE gene"] == "no ASE"]["compare_2allele"] == "ALT"
    ).sum()
    #           ASE   no ASE\n",
    # REF>ALT    13      8\n",
    # REF<=ALT   14      12\n",
    table = np.array([[ASE_REF, noASE_REF], [ASE_ALT, noASE_ALT]])
    oddsr, p = fisher_exact(table, alternative="two-sided")
    return p, ASE_REF, ASE_ALT, noASE_REF, noASE_ALT


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.error("Gene candidate argument missing.")
        sys.exit(1)

    gene_candidate = sys.argv[1]
    gene_name = sys.argv[2]

    cutoff_totalcount = 10
    cutoff_nind = 1
    data_dir = "/data2/1000Genome"
    save_dir = "/data2/genes"

    output_dir = Path(os.path.join(save_dir, gene_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    #######################################
    ####################################### step3: haplotype identification
    #######################################
    common_SNP_file = output_dir / "common_SNP_composition.tsv"
    common_SNP_frequency_file = output_dir / "common_SNP_frequency_table.tsv"
        ## (1) common SNP
    common_SNP = collect_info_one_gene(cutoff_totalcount, data_dir, gene_candidate, common_SNP_file)
        ## (2) collect all SNP combination frequency information

    common_SNP_data_snp, common_SNP_data_snp_list = collect_info_common_SNPs(
        common_SNP , common_SNP_frequency_file
    )
    n_total_ind_option2 = len(pd.unique(common_SNP["sample"]))

    get_haplotype(
        common_SNP, 
        common_SNP_data_snp_list, 
        common_SNP_data_snp, 
        n_total_ind_option2, 
        gene_name, 
        save_dir
    )
    
    print("geneID\tchr\tpos\ttotal_individual\tn_individual\tpercent_individual\tpval\tASE_REF\tASE_ALT\tnoASE_REF\tnoASE_ALT\n")
    for pos in data_snp_list:
        if data_snp.to_dict()[pos] >= cutoff1:
            count = data_snp.to_dict()[pos]
            chrN = data["chr"].iloc[0]
            (
                p,
                ASE_REF,
                ASE_ALT,
                noASE_REF,
                noASE_ALT,
            ) = calculate_2sidedfisherexact_variant(pos, data)
            info = [
                [
                    gene_name,
                    chrN,
                    pos,
                    total,
                    count,
                    round(count/total,2),
                    round(p, 3),
                    ASE_REF,
                    ASE_ALT,
                    noASE_REF,
                    noASE_ALT,
                ]
            ]
            for r in info:
                print("\t".join(map(str, r)))
